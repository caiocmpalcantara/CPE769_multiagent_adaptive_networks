classdef N_Gaussian < Noise
    properties
        sigma
        mu
    end

    methods
        function obj = N_Gaussian(varargin)
            p = inputParser;

            default_sigma = 1;
            check_sigma = @(x) isnumeric(x) && (imag(x)==0);

            default_mu = 0;
            check_mu = @(x) isnumeric(x) && (imag(x)==0);

            addOptional(p, 'sigma', default_sigma, check_sigma);
            addOptional(p, 'mu', default_mu, check_mu);

            try
                parse(p, varargin{:});
                obj.sigma = p.Results.sigma;
                obj.mu = p.Results.mu;

            catch exception
                error('An error occurred: %s\n', exception.message);
            end

        end
        function value = realize(obj, size_)
            try
                value = obj.sigma * randn(size_n) + obj.mu;
            catch exception
                error('An error occurred: %s\n', exception.message);
            end
        end
    end
end 