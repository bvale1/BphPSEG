import numpy as np
import torch
import logging
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO)

class feature_extractor():

    def __init__(self, data, roi=None, mask=None):
                
        assert isinstance(data, (np.ndarray, torch.Tensor)), \
            'data must be a numpy array or torch tensor'
        assert (len(data.shape) == 3), 'data must be of shape (npulses, x, z)'
        if type(np.ndarray):
            data = torch.from_numpy(data)            
            
        # from shape (npulses, Nx, Nz) to (npulses, Nx*Nz)
        self.image_size = data.shape[1:]
        self.data = torch.flatten(data, start_dim=1, end_dim=2)
        self.npulses = self.data.shape[0]
        self.n = torch.arange(
            self.npulses, dtype=torch.float32, requires_grad=False
        ).unsqueeze(0)
        
        # roi defines a circular region of interest
        if roi:
            assert isinstance(roi, list, np.ndarray, tuple, torch.Tensor), \
                'roi must be a list, numpy array, tuple or torch tensor'
            assert len(roi) == 3, 'roi must have 3 elements (x,z,r) in pixels' 
            
            [X, Z] = torch.meshgrid(
                torch.arange(data.shape[1]),
                torch.arange(data.shape[2])
            )
            R = torch.sqrt((X-roi[0])**2 + (Z-roi[1])**2)
            self.mask = torch.flatten(R <= roi[2])
            self.data = self.data[:, self.mask]
        elif mask is not None:
            assert isinstance(mask, (np.ndarray, torch.Tensor)), \
                'mask must be a numpy array or torch tensor'
            if type(mask) == np.ndarray:
                mask = torch.from_numpy(mask)
            assert mask.shape == self.image_size, \
                'mask must have the same shape as data xz dimensions'
            self.mask = torch.flatten(mask)
            self.data = self.data[:, self.mask]
        else:
            self.mask = None
    
        # feature extraction is on a pixel by pixel basis
        # so xz dimensions must be leading (batch) dimensions
        # (npulses, npixels) -> (npixels, npulses)
        self.data = self.data.T
        self.features = {'A': None, 'k': None, 'b': None, 'R_sqr': None}
        self.diverages_mask = None
        
    def as_images(self):
        # reshapes features from (Nx*Nz) to (Nx, Nz)
        if isinstance(self.mask, torch.Tensor) and not torch.all(self.mask).item():
            for arg in self.features.keys():
                if self.features[arg] is None:
                    continue
                # areas outside of roi are set to nan
                feature = (torch.nan*torch.empty(
                    self.image_size[0]*self.image_size[1],
                    dtype=torch.float32,
                    requires_grad=False
                ))
                feature[self.mask] = self.features[arg]
                self.features[arg] = feature.reshape(self.image_size)
        else:
            for arg in self.features.keys():
                if self.features[arg] is None:
                    continue
                self.features[arg] = self.features[arg].reshape(self.image_size)
                
    def flatten_features(self):
        for arg in self.features.keys():
            if self.features[arg] is None:
                continue
            self.features[arg] = self.features[arg].flatten()
            if isinstance(self.mask, torch.Tensor) and not torch.all(self.mask).item():
                self.features[arg] = self.features[arg][self.mask]        
        
    
    def get_features(self, asTensor=True):
        # can return features as either a dictionary or a tensor
        if asTensor:
            for arg in self.features.keys():
                if self.features[arg] is None:
                    del self.features[arg]
            tensor = torch.empty(
                (len(self.features.keys()), self.image_size[0], self.image_size[1]),
                dtype=torch.float32,
                requires_grad=False
            )
        # reshapes features from (Nx*Nz) to (Nx, Nz) before returning
        if isinstance(self.mask, torch.Tensor) and not torch.all(self.mask).item():
            for i, arg in enumerate(self.features.keys()):
                # areas outside of mask are set to nan
                feature = (torch.nan*torch.empty(
                    self.image_size[0]*self.image_size[1],
                    dtype=torch.float32,
                    requires_grad=False
                ))
                feature[self.mask] = self.features[arg]
                if asTensor:
                    tensor[i] = feature.reshape(self.image_size)
                else:
                    self.features[arg] = feature.reshape(self.image_size)
                
        else:
            for i, arg in enumerate(self.features.keys()):
                if asTensor:
                    tensor[i] = self.features[arg].reshape(self.image_size)
                else:
                    self.features[arg] = self.features[arg].reshape(self.image_size)
        
        if asTensor:
            return tensor, self.features.keys()
        else:
            return self.features, self.features.keys()
    
    
    def fft_exp_fit(self):
        # compute fast fourier transfrom of pixel vectors
        # use this to compute starting values for exponential fit
        # then use NLS_GN_exp_fit find the best fit
        
        # Andrei A. istratov and Oleg F. Vyvenko (1998) Exponential analysis in physical phenomena
        # https://pubs.aip.org/aip/rsi/article/70/2/1233/438854/Exponential-analysis-in-physical-phenomena
        # https://doi.org/10.1063/1.1149581
        print('initializing exponential fit parameters via fourier method of transients')
        fft = torch.fft.fft(self.data, dim=0)
        logging.debug(f'fft: {fft.shape}, {fft.dtype}')
        # compute angular frequency of components
        omega = 2 * np.pi * self.n / (self.npulses-1)
        logging.debug(f'omega: {omega.shape}, {omega.dtype}')
        
        k = - omega[0,1] * torch.real(fft[:,0]) / torch.imag(fft[:,1])
        A = (omega[0,1]**2 + k**2) * torch.real(fft[:,1]) / (k * (1-torch.exp(-k*self.npulses)))
        b = (torch.real(fft[:,0])/self.npulses) - ((A/(k*self.npulses)) * (1-torch.exp(-k*self.npulses)))
        
        self.features['A'] = A
        self.features['k'] = k
        self.features['b'] = b
        self.diverages_mask = None
    
    
    def NLS_GN_exp_fit(self,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       maxiter=10):
        
        assert isinstance(self.features['A'], torch.Tensor) \
             and isinstance(self.features['k'], torch.Tensor) \
             and isinstance(self.features['b'], torch.Tensor), \
            'fft_exp_fit must be performed first'
        
        # implementation of the Guass-Newton non-linear least
        # squares method for each pixel vector in parallel
        
        print('gauss newton method: maxiter={}'.format(maxiter))
                
        # negative values of k cause the fit to diverge
        self.features['k'][self.features['k']<0.0] = 0.0
        
        # beta[0] = A; beta[1] = k; beta[2] = b
        # beta.shape = (npixels, 3, 1)
        beta = torch.cat(
            (
                self.features['A'].unsqueeze(-1),
                self.features['k'].unsqueeze(-1),
                self.features['b'].unsqueeze(-1)
            ), dim=-1
        ).to(dtype=torch.float64).unsqueeze(-1)
        beta.requires_grad = False
        beta = beta.to(device)
        
        # float64 used for addional stability
        self.data = self.data.to(device, dtype=torch.float64)
        
        n = self.n.clone().to(device, dtype=torch.float64)
        
        # set inf and nan values to 0.0
        beta[torch.logical_not(torch.isfinite(beta))] = 0.0
        
        # the fit will still diverge for some pixels
        # a mask is used to ignore these pixels, then the slower but more robust
        # scipy.optimize.curve_fit is used to fit these pixels
        diverages_mask = torch.ones_like(self.features['A'], dtype=torch.bool)
        # 0 = divergence, 1 = convergence of the fit
        
        # Jacobain.shape = (npixels, npulses, 3)
        J = torch.ones(
            (self.data.shape[0], self.npulses, 3),
            dtype=torch.float64,
            requires_grad=False,
            device=device,
        )
        
        for i in range(maxiter):
            logging.debug(f'iteration {i+1}/{maxiter}')
            
            # compute residuals
            r = (self.data - beta[:,0,:]*torch.exp(-beta[:,1,:]*n) - beta[:,2,:]).unsqueeze(-1)
            logging.debug(f'r.shape: {r.shape}, {r.dtype}')
            
            # mask diverging parameters
            # all parameters along axis 1 (A, k or b) must be finite
            diverages_mask = torch.logical_and(
                diverages_mask, 
                torch.all(torch.isfinite(beta.squeeze()), dim=1)
            )
            
            # compute gradient (Jacobian)
            logging.debug(f'beta: {beta.shape}, {beta.dtype}')
            logging.debug(f'n: {n.shape}, {n.dtype}')
            J[:, :, 0] = torch.exp(-beta[:,1,:]*n)
            J[:, :, 1] = (-beta[:,0,:]*n*torch.exp(-beta[:,1,:]*n))
            # gradient of f with respect to b (J[:, :, 2]) is always 1.0
            
            logging.debug(f'J: {J.shape}, {J.dtype}')
            
            # similarly mask diverging gradients
            diverages_mask = torch.logical_and(
                diverages_mask,
                torch.all(
                    torch.all(torch.isfinite(J), dim=2), dim=1
                )
            )
            
            logging.debug('check for diverging fit parameters or gradients')
            logging.debug(f'diverages_mask: {torch.sum(diverages_mask, dtype=torch.int32)}')
            logging.debug(f'max(beta) {torch.max(beta[diverages_mask]).item()}')
            logging.debug(f'min(beta) {torch.min(beta[diverages_mask]).item()}')
            logging.debug(f'max(J) {torch.max(J[diverages_mask]).item()}')
            logging.debug(f'min(J) {torch.min(J[diverages_mask]).item()}')
            
            # compute GN step, by multiplying residuals by moor penrose inverse
            step = torch.linalg.lstsq(
                J[diverages_mask],
                r[diverages_mask]
            ).solution
                
            logging.debug(f'step {step.shape}, {step.dtype}')
            beta[diverages_mask] = (beta[diverages_mask] + step)

       
        self.diverages_mask = diverages_mask
        beta = beta.to(torch.device('cpu'))
        self.features['A'] = beta[:,0,0].to(dtype=torch.float32)
        self.features['k'] = beta[:,1,0].to(dtype=torch.float32)
        self.features['b'] = beta[:,2,0].to(dtype=torch.float32) 
        self.data = self.data.to(torch.device('cpu'), dtype=torch.float32)
        
        
    def NLS_scipy(self, display_progress=False, scale_data=False):
        
        if scale_data:
            logging.info('warning: scaling data is generally detrimental to the fit')
        
        if not isinstance(self.features['A'], torch.Tensor) \
             or not isinstance(self.features['k'], torch.Tensor) \
             or not isinstance(self.features['b'], torch.Tensor):
                 
            logging.info('fit parameters not found, initializing...')
            self.features['A'] = torch.ones(
                self.data.shape[0], dtype=torch.float32, requires_grad=False
            )
            self.features['k'] = torch.ones(
                self.data.shape[0], dtype=torch.float32, requires_grad=False
            )
            self.features['b'] = torch.ones(
                self.data.shape[0], dtype=torch.float32, requires_grad=False
            )
            
        if isinstance(self.diverages_mask, torch.Tensor):
            if torch.all(self.diverages_mask).item():
                logging.info('all pixels converged, aborting scipy fit')
                return None
            else:
                self.diverages_mask = self.diverages_mask.numpy()
        else:
            # GN_NLS_exp_fit has not been performed
            self.diverages_mask = np.zeros_like(
                self.features['A'], dtype=np.bool
            )
        
        # scipy.optimize.least_squares levenberg–marquardt method is more robust
        # but slow and not parallised, it is used to fit 
        # pixels that diverge in NLS_GN_exp_fit
        residuals = lambda x, y, n: y - x[0] * np.exp(-x[1] * n) - x[2]
    
        # negative values of k cause the fit to diverge
        self.features['k'][self.features['k']<0.0] = 0.0
        
        beta = np.concatenate(
            (
                self.features['A'].unsqueeze(-1).numpy(),
                self.features['k'].unsqueeze(-1).numpy(),
                self.features['b'].unsqueeze(-1).numpy()
            ), axis=-1
        )
        
        # set inf and nan values to zero
        beta[np.logical_not(np.isfinite(beta))] = 0.0
        
        self.data = self.data.numpy()
        n = np.arange(self.npulses)
        n_pixels = np.sum(np.logical_not(self.diverages_mask))
        
        for i, x in enumerate(self.data[np.logical_not(self.diverages_mask)]):
            if (i+1)%2000 == 0 and display_progress:
                logging.info(f'{round((i+1)*100/n_pixels, 2)}%')
                
            if scale_data:
                scaler = RobustScaler()
                x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
                
            beta[i] = least_squares(
                residuals, 
                x0=beta[i,:],
                args=(x, n),
                method='lm',
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9
            ).x
            
            if scale_data:
                beta[i] = scaler.inverse_transform(beta[i].reshape(-1, 1)).flatten()
        
        self.data = torch.from_numpy(self.data)
        self.features['A'] = torch.from_numpy(beta[:,0])
        self.features['k'] = torch.from_numpy(beta[:,1])
        self.features['b'] = torch.from_numpy(beta[:,2])

    
    def R_squared(self):
        
        assert isinstance(self.features['A'], torch.Tensor) \
             and isinstance(self.features['k'], torch.Tensor) \
             and isinstance(self.features['b'], torch.Tensor), \
            'fft_exp_fit must be performed first'
    
        SS_res = torch.sum(
            (self.data - self.features['A'].unsqueeze(-1) * 
             np.exp(-self.features['k'].unsqueeze(-1) *
                self.n) - self.features['b'].unsqueeze(-1))**2,
            dim=-1
        )
        SS_tot = torch.sum(
            (self.data - torch.mean(
                self.data, dim=-1, keepdim=True
            )
        )**2, dim=-1)
        
        self.features['R_sqr'] = 1 - (SS_res / SS_tot)
        self.features['R_sqr'][torch.logical_not(torch.isfinite(self.features['R_sqr']))] = 0.0
        self.features['R_sqr'][self.features['R_sqr'] < 0.0] = 0.0
        
    
    def filter_features(self, mask=None, threshold=(0.5, 0.5), filter_size=3):
        self.as_images()
        for arg in self.features.keys():
            if self.features[arg] is None:
                continue
            self.features[arg] = self.masked_filter(
                self.features[arg], 
                mask=mask,
                threshold=threshold,
                filter_size=filter_size
            )
        self.flatten_features()
            
        
    def masked_filter(self, img, mask=None, threshold=(0.5, 0.5), filter_size=3):
        # mask refers to the background mask
        # since background pixels should not contribute to the quantiles
        # filter_mask is the pixels to be filtered
        
        assert len(img.shape) == 2, 'img must be 2D'
        img[torch.isinf(img)] = torch.nan
        
        # apply median filter to remove "salt and pepper noise"
        # https://medium.com/@florestony5454/median-filtering-with-python-and-opencv-2bce390be0d1
        if mask is None:
            mask = torch.ones_like(img, dtype=torch.bool, requires_grad=False)
        else:
            if type(mask) == np.ndarray:
                mask = torch.from_numpy(mask)
            assert img.shape == mask.shape, 'img and mask must have the same shape'
            assert mask.dtype == bool or mask.dtype == torch.bool, 'mask must be a boolean tensor'
        
        # filter a pixel if it is nan and it is not a background pixel
        filter_mask = torch.logical_and(torch.isnan(img), mask)
        
        if threshold:
            top_quantile = torch.nanquantile(img[mask], threshold[1])
            bottom_quantile = torch.nanquantile(img[mask], threshold[0])
            filter_mask[mask] = torch.logical_or(img<bottom_quantile, filter_mask)[mask]
            filter_mask[mask] = torch.logical_or(img>top_quantile, filter_mask)[mask]
                        
        filtered_img = torch.from_numpy(
            median_filter(
                img.numpy(), 
                size=filter_size
            )
        )
        img[filter_mask] = filtered_img[filter_mask]
        
        return img
            
    def radial_distance(self, dx):
        # the distance from the centre of the image of each pixel
        # in units of mm
        [X, Y] = torch.meshgrid(
            torch.arange(self.image_size[0]) - self.image_size[0]/2, 
            torch.arange(self.image_size[1]) - self.image_size[1]/2
        )
        self.features['radial_distance'] = torch.sqrt(X**2 + Y**2).flatten()[self.mask] * dx
        
    def threshold_features(self):
        data_max = torch.max(self.data)
        for arg in self.features.keys():
            if self.features[arg] is None:
                continue
            if arg == 'A' or arg == 'b':
                self.features[arg][self.features[arg] < -data_max] = -data_max
                self.features[arg][self.features[arg] > data_max] = data_max
            elif arg == 'k':
                self.features[arg][self.features[arg] < -0.5] = -0.5
                self.features[arg][self.features[arg] > 0.5] = 0.5
        
    def normalise(self):
        # normalise data so that each pixel vector has a maximum value of 1.0
        data_max = torch.max(self.data, dim=1, keepdim=True)[0]
        # avoid division by zero
        self.data = self.data / (data_max + 1e-8)
        
    def differetial_image(self):
        # compute the difference between the first and last pulse
        self.features['diff'] = (self.data[:, 0] - self.data[:, -1])
        
    def range_image(self):
        # compute the range of the image
        self.features['range'] = (
            torch.max(self.data, dim=1)[0] - torch.min(self.data, dim=1)[0]
        )