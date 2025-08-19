classdef Linear_Obs < Sensor_Model
    properties (Access = protected)
        R
        H
    end

    methods
        function obj = Linear_Obs(varargin)
            % Must capture the 'dim' varargin
            obj@Sensor_Model(varargin);
            
            p = inputParser;

            % Defining mandatory arguments             

            % Defining optional arguments
            default_R_matrix = zeros(obj.dim);
            check_R_matrix = @(x) isnumeric(x) && any(size(x)==[obj.dim, obj.dim]);
            addOptional(p, 'R_matrix', default_R_matrix, check_R_matrix)

            default_H_matrix = eye(obj.dim);
            check_H_matrix = @(x) isnumeric(x) && any(size(x)==[obj.dim, obj.dim]);
            addOptional(p, 'H_matrix', default_H_matrix, check_H_matrix);
           
            parse(p, varargin{:});
            try
                R = p.Results.R_matrix;
                H = p.Results.H_matrix;

                % Check the dimensions
                if ~any(strcmp('R_matrix', p.UsingDefaults)) && any(size(R) ~= obj.dim) 
                    error('R_matrix has wrong number of dimension.')
                elseif ~any(strcmp('H_matrix', p.UsingDefaults)) && any(size(H) ~= obj.dim)
                    error('H_matrix has wrong number of dimension.')
                end
            
                obj.R = R;
                obj.H = H;
            catch exception
                error('An error occurred: %s.\n', exception.message)
            end
            
        end

        % function Pa = Pa_init(obj, varargin)
        %     p = inputParser;
            
        %     default_delta = 1e-1;
        %     check_delta = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0);
        %     addOptional(p, 'delta', default_delta, check_delta);
            
        %     parse(p, varargin{:});
        %     try
        %         Pa = p.Results.delta * eye(obj.dim);
        %     catch exception
        %         error('An error occurred: %s.\n', exception.message);
        %     end    
        % end
        
        % function xa = xa_init(obj, varargin)
        %     p = inputParser;

        %     default_initial_state = zeros(obj.dim, 1);
        %     check_initial_state = @(x) isnumeric(x) && (length(x)==obj.dim);
        %     addOptional(p, 'initial_state', default_initial_state, check_initial_state);
            
        %     parse(p, varargin{:});
        %     try
        %         % if ~any(strcmp('initial_state', p.UsingDefaults)) && (length(p.Results.initial_state) ~= obj.dim)
        %         %     error('The initial state used is different than model dimension.')
        %         % end
        %         xa = p.Results.initial_state;
        %     catch exception
        %         error('An error occurred: %s.\n', exception.message);
        %     end
        % end

        function R = update_R(obj, varargin)
            R = obj.R;
        end
        
        function H = update_H(obj, varargin)
            H = obj.H;
        end
    end
end