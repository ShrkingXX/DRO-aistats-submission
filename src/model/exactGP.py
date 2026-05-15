import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel="rbf", lengthscale_prior=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel == "matern":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel == "rq":
            self.covar_module = gpytorch.kernels.RQKernel()
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
            
        if lengthscale_prior:
            self.covar_module.lengthscale = lengthscale_prior
        
        self.covar_module.register_constraint("raw_lengthscale", 
                                             gpytorch.constraints.Interval(0.1, 10.0))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
