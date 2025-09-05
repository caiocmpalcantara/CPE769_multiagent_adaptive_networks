classdef Rls2 < Agent_technique2
    %{ Rest Leasts Square (RLS) Algorithm
    %  Based on Adaptive Filtering Algorithms and Practical Implementation, Chap.5, Diniz 
    %}
    properties (Access=public)
        P                   % estimate error covariance matrix
        k                   % gain vector (literature: similar to Kalman gain)
        alpha               % innovation vector
        H                   % Observational model matrix (obs: u^T = H) 
        lambda              % Forgetting factor
        start_vals struct   % Initial value 
        iteracts            % Number of iteractions (time steps) 

    end
    methods (Access=public)
        function obj = Rls2(varargin)
            DEBUG(varargin)
            obj@Agent_technique2(varargin{:});

            p = inputParser;
            p.KeepUnmatched = true;

            addParameter(p, 'H_matrix', ones(obj.y_dim, obj.x_dim), @(x) isnumeric(x) && (all(size(x)==[obj.y_dim, obj.x_dim])));
            addParameter(p, 'lambda', 0.9, @(x) isscalar(x) && (x>0));
            addParameter(p, 'start_vals', )

        end
    end
end