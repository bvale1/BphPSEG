import numpy as np
import torch


class feature_extractor():

    def __init__(self, data, roi=None):
        
        
        assert isinstance(data, np.ndarray, torch.Tensor), \
            'data must be a numpy array or torch tensor'
        assert (len(data.shape) == 3), 'data must be of shape (npulses, x, z)'
        if type(np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, requires_grad=False)
            
            
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
    
        # feature extraction is on a pixel by pixel basis
        # so xz dimensions must be leading (batch) dimensions
        # (npulses, npixels) -> (npixels, npulses)
        self.data = self.data.T
        self.features = {}
        
    
    def get_features(self):
        # reshapes features from (Nx*Nz) to (Nx, Nz) before returning
        for arg in self.features.keys():
            self.features[arg] = self.features[arg].reshape(self.image_size)
            
        return self.features
    
    
    def fft_exp_fit(self):
        # compute fast fourier transfrom of pixel vectors
        # use this to compute starting values for exponential fit
        # then use NLS_GN_exp_fit find the best fit
        
        # Andrei A. istratov and Oleg F. Vyvenko (1998) Exponential analysis in physical phenomena
        # https://pubs.aip.org/aip/rsi/article/70/2/1233/438854/Exponential-analysis-in-physical-phenomena
        # https://doi.org/10.1063/1.1149581
        print('initializing exponential fit parameters via fourier method of transients')
        fft = torch.fft.fft(self.data, dim=0)
        print('fft.shape')
        print(fft.shape)
        # compute angular frequency of components
        omega = 2 * np.pi * self.n / (self.n_max-1)
        print('omega.shape')
        print(omega.shape)
        
        k = - omega[0,1] * torch.real(fft[:,0]) / torch.imag(fft[:,1])
        A = (omega[0,1]**2 + k**2) * torch.real(fft[:,1]) / (k * (1-torch.exp(-k*self.n_max)))
        b = (torch.real(fft[:,0])/self.n_max) - ((A/(k*self.n_max)) * (1-torch.exp(-k*self.n_max)))
        
        self.features['A'] = A
        self.features['k'] = k
        self.features['b'] = b
    
    
    def NLS_GN_exp_fit(self,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                       maxiter=10):
        
        assert isinstance(self.features['A'], self.features['k'], self.features['b']), \
            'fft_exp_fit must be performed first'
        
        # implementation of the Guass-Newton non-linear least
        # squares method for each pixel vector in parallel
        
        print('gauss newton method: maxiter={}'.format(maxiter))
                
        # negative values of k cause the fit to diverge
        self.k[self.k<0.0] = 0.0
        
        # beta[0] = A; beta[1] = k; beta[2] = b
        beta = torch.cat(
            (
                self.features['A'].unsqueeze(-1),
                self.features['k'].unsqueeze(-1),
                self.features['b'].unsqueeze(-1)
            ), dim=-1
        ).to(dtype=torch.float64).unsqueeze(-1)
        beta.requires_grad = False
        beta = beta.to(device)
        print('beta.shape')
        print(beta.shape)
        
        # float64 used for addional stability
        self.data = self.data.to(device, dtype=torch.float64)
        
        n = self.n.clone().to(device, dtype=torch.float64)
        
        # set inf and nan values to 0.0
        beta[torch.logical_not(torch.isfinite(beta))] = 0.0
        
        # the fit will still diverge for some pixels
        # a mask is used to ignore these pixels, then the slower but more robust
        # scipy.optimize.curve_fit is used to fit these pixels
        diverages_mask = torch.zeros_like(self.A, dtype=torch.bool)
        
        df_db = torch.ones(
            (self.data.shape[0], self.n_max, 1),
            dtype=torch.float64, 
            requires_grad=False,
            device=device
        )
        
        J = torch.ones(
            (self.data.shape[0], self.n_max, 3),
            dtype=torch.float64,
            requires_grad=False,
            device=device,
        )
        
        for i in range(maxiter):
            
            # compute residuals
            r = self.data - beta[:,0,:]*torch.exp(-beta[:,1,:]*n) - beta[:,2,:]
            #print('r.shape')
            #print(r.shape)
            #print((r[100,150]))
            # r.shape = torch.Size([self.len, 15])
            #print(beta[100,150,:,:])
            #print(beta[:,:,0].min())
            #print(beta[:,:,0].max())
            #print(beta[:,:,1].min())
            #print(beta[:,:,1].max())
            #print(beta[:,:,2].min())
            #print(beta[:,:,2].max())
            
            #beta[:,:,0,0][beta[:,:,0,0]<0.0] = 0.0
            #beta[:,:,1,0][beta[:,:,1,0]<0.0] = 0.0
            
            # compute gradient (Jacobian)
            print(n.shape)
            print(beta.shape)
            J = torch.cat(
                (
                    torch.exp(-beta[:,1,:]*n).unsqueeze(-1),
                    (-beta[:,0,:]*n*torch.exp(-beta[:,1,:]*n)).unsqueeze(-1),
                    df_db
                ), dim = -1
            )
            J[torch.logical_not(torch.isfinite(J))] = 0.0
            
            print(J.min())
            print(J.max())
            
            print('J.shape')
            print(J.shape)
            print(torch.logical_not(torch.isfinite(J)))
            
            print(beta[J[:,:,1]==torch.inf])
            print(beta[J[:,:,1]==torch.nan])
            print(J[J==torch.inf])
            print(J[J==torch.nan])
            
            print(J)
            # compute GN step, by multiplying residuals by moor penrose inverse
            '''
            step = torch.matmul(
                torch.linalg.pinv(J.double()).to(dtype=torch.float32),
                r.unsqueeze(-1)
            )
            '''
            step = torch.linalg.lstsq(J, r.unsqueeze(-1)).solution
            print('step.shape')
            print(step.shape)
            #print(step[100,150])
            beta[torch.logical_not(diverages_mask)] = (beta + step)[torch.logical_not(diverages_mask)]

            #print(beta[100,150,:,:])
            diverages_mask += torch.logical_not(torch.isfinite(beta))      
        
        
        beta = beta.to(torch.device('cpu'))
        self.A = beta[:,0,0].to(dtype=torch.float32)
        self.k = beta[:,1,0].to(dtype=torch.float32)
        self.b = beta[:,2,0].to(dtype=torch.float32) 
        self.data = self.data.to(torch.device('cpu'), dtype=torch.float32)
        
    
    def R_squared(self):
        
        assert isinstance(self.A, self.k, self.b), 'expontential fit must be performed first'
    
        SS_res = torch.sum(
            (self.data - self.A.unsqueeze(-1) * np.exp(-self.k.unsqueeze(-1) *
                self.n.unsqueeze(0)) - self.b.unsqueeze(-1))**2,
            dim=-1
        )
        SS_tot = torch.sum(
            (self.data - torch.mean(
                self.data, dim=-1, keepdim=True
            )
        )**2, dim=-1)
        
        self.R_sqr = 1 - (SS_res / SS_tot)
        
        