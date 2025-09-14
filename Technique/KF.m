classdef KF < Kalman
    % Class: KF - Represent an agent technique that implements Linear KF
    % This class is derived from Kalman which is derived from Agent_technique
    % in order to implement a Linear (Sensor Model) Kalman Filtering technique
    % for multiagents systems. It implements  
    properties
        H
        R
        sigma
    end

    methods
        function obj = KF(varargin)
            DEBUG(varargin)
            obj@Kalman(varargin{:});
            DEBUG(varargin)

            p = inputParser;
            p.KeepUnmatched = true;
            
            default_H_matrix = eye(obj.y_dim, obj.x_dim);
            check_H_matrix = @(x) isnumeric(x) && all(size(x)==[obj.y_dim, obj.x_dim]);
            addOptional(p, 'H_matrix', default_H_matrix, check_H_matrix);
            
            default_R_matrix = eye(obj.y_dim);
            check_R_matrix = @(x) isnumeric(x) && all(size(x)==[obj.y_dim obj.y_dim]);
            addOptional(p, 'R_matrix', default_R_matrix, check_R_matrix);

            default_sigma = 1;
            check_sigma = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addOptional(p, 'sigma', default_sigma, check_sigma);

            try
                parse(p, varargin{:});

                obj.H = p.Results.H_matrix;
                if ~any(strcmp('R_matrix', p.UsingDefaults)) && any(strcmp('sigma', p.UsingDefaults))
                    obj.R = p.Results.R_matrix;
                    obj.sigma = [];
                elseif any(strcmp('R_matrix', p.UsingDefaults)) && ~any(strcmp('sigma', p.UsingDefaults))
                    obj.sigma = p.Results.sigma;
                    obj.R = obj.sigma^2 * eye(obj.y_dim);
                else
                    obj.R = p.Results.R_matrix;
                    obj.sigma = sqrt(obj.R(1,1));
                end
                obj.R = p.Results.R_matrix;

            catch exception
                error('An error occurred: %s', exception.message);
            end

            DEBUG(obj.H)
            DEBUG(obj.R)
            DEBUG(obj.sigma)

        end

        % Method: update_obs - Update the observation sample in order to process the new sample
        % -------------------------------------
        function y = update_obs(obj, measurement)
            y = zeros(obj.y_dim, 1);
            try
                switch class(measurement)
                    case 'GeoPoint'
                        y(1) = measurement.get_cart(1).get_m();
                        y(2) = measurement.get_cart(2).get_m();
                    
                    case 'Displacement'
                        y(1) = measurement.get_cart(1).get_m();
                        y(2) = measurement.get_cart(2).get_m();

                    case 'double'
                        if length(measurement) == obj.y_dim
                            y = measurement;
                            y = reshape(y, obj.y_dim, 1);
                        else
                            error('KF: Wrong dimension in measurement type: it must agree with obj.y_dim')
                        end
                    
                    % case ''
                    %     if length(measurement) == obj.y_dim
                    %         y = measurement;
                    %         y = reshape(y, obj.y_dim, 1);
                    %     end
                    
                    otherwise
                        error('KF: Unknow measurement type.')
                end
            catch exception
                error("An error occurred: %s", exception.message);
            end
        end
        
        % Method: update_y_hat - Based on the prior estimate, update the new expected observation
        % ----------------------------------
        function y_hat = update_y_hat(obj, varargin) 
            y_hat = obj.update_H(varargin{:}) * obj.xa_hat; %TODO: Verify
        end
        
        % Method: update_Py - Based on the prior P matrix, update the new Py matrix
        % ---------------------------------
        function Py = update_Py(obj, varargin)
            DEBUG(varargin)
            Py = obj.H * obj.Pa * obj.H';
        end

        % Method: update_Pxy - Based on the prior P matrix, update the new Pxy matrix
        % ---------------------------------
        function Pxy = update_Pxy(obj, varargin)
            Pxy = obj.Pa * obj.H';
        end

        % Method: update_Px - Based on the posterior P matrix, update the new prior P matrix
        % ------------------------------------
        function R = update_R_matrix(obj, varargin)
            
            p = inputParser;
            p.KeepUnmatched = true;

            DEBUG(varargin)
            
            default_R_matrix = obj.R;
            DEBUG(obj.R)
            DEBUG(obj.y_dim)
            check_R_matrix = @(x) isnumeric(x) && all(size(x)==[obj.y_dim obj.y_dim]);
            addOptional(p, 'R_matrix', default_R_matrix, check_R_matrix);

            default_sigma = 1;
            check_sigma = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addOptional(p, 'sigma', default_sigma, check_sigma);

            try
                parse(p, varargin{:});

                if ~any(strcmp('R_matrix', p.UsingDefaults))
                    obj.R = p.Results.R_matrix;
                elseif ~any(strcmp('sigma', p.UsingDefaults)) && any(strcmp('R_matrix', p.UsingDefaults))
                    obj.sigma = p.Results.sigma;
                    obj.R = obj.sigma^2 * eye(obj.y_dim);
                end
                R = obj.R;
            catch exception
                error('An error occurred: %s', exception.message);
            end
        end

        % Method: get_H - get the H matrix to the user.
        % ------------------------------------
        function H = get_H(obj)
            H = obj.H;
        end

        % Method: update_H - Update the new prior P matrix
        % ------------------------------------
        function H = update_H(obj, varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            
            default_H_matrix = obj.H;
            check_H_matrix = @(x) isnumeric(x) && all(size(x)==[obj.y_dim, obj.x_dim]);
            addOptional(p, 'H_matrix', default_H_matrix, check_H_matrix);
            
            try
                parse(p, varargin{:});
                obj.H = p.Results.H_matrix;
                H = obj.H; 
            catch exception
                error('KF: An error occurred: %s', exception.message); 
            end
        end

         function obj = update_params(obj, varargin)
            % @brief Update the Kalman filtering parameters based on new information.
           
            try
                update_params@Kalman(obj, varargin{:});
                
                ind = find(strcmp(varargin, 'H_matrix'));
                if ~isempty(ind)
                    obj.H = varargin{ind+1};
                end

            catch exception
                error('KF: An error occurred: %s', exception.message)
            end
        end

        % TODO: Create a method get_params to get the H matrix.

        % Method: reset - Reset the technique
        % ------------------------------------
        % function obj = reset(obj)
        %     % Dummy for a while
        %     obj.Pp = obj.system_model.Pa_init('delta', obj.system_model.initial_params.delta);
        %     obj.Pa = obj.Pp;
        %     obj.xa_hat = obj.system_model.xa_init('initial_state', obj.system_model.initial_params.initial_state);
        %     obj.xp_hat = obj.xa_hat;
        % end
    end
end