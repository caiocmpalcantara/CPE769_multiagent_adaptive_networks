classdef Linear_State < System_Model
    properties (Access = protected)
        Q
        A
        initial_params struct
    end

    methods
        
        function obj = Linear_State(varargin)

            % Must capture the 'dim' varargin
            % if DEBUG == 1
            %     disp('### DEBUG ### Linear_State::Linear_State')
            %     disp('varargin')
            %     disp(varargin)
            %     ind = find(strcmp(varargin, 'dim'));
            %     if ~isempty(ind)
            %         disp(varargin{ind+1})                
            %     end
            % end
            % DEBUG(varargin)
            obj@System_Model(varargin{:});

            % if debug
                
            % end
            % DEBUG(varargin)
            
            p = inputParser;
            p.KeepUnmatched = true;

            % Defining mandatory arguments             

            % Defining optional arguments
            default_Q_matrix = zeros(obj.dim);
            DEBUG(obj.dim)
            check_Q_matrix = @(x) isnumeric(x) && all(size(x)==[obj.dim, obj.dim]);
            addOptional(p, 'Q_matrix', default_Q_matrix, check_Q_matrix)

            default_A_matrix = eye(obj.dim);
            check_A_matrix = @(x) isnumeric(x) && all(size(x)==[obj.dim, obj.dim]);
            addOptional(p, 'A_matrix', default_A_matrix, check_A_matrix);
           
            parse(p, varargin{:});
            try
                Q = p.Results.Q_matrix;
                A = p.Results.A_matrix;

                % Check the dimensions
                if ~any(strcmp('Q_matrix', p.UsingDefaults)) && any(size(Q) ~= obj.dim) 
                    error('Q_matrix has wrong number of dimension.')
                elseif ~any(strcmp('A_matrix', p.UsingDefaults)) && any(size(A) ~= obj.dim)
                    error('A_matrix has wrong number of dimension.')
                end
            
                obj.Q = Q;
                obj.A = A;
                
                obj.initial_params = struct();
                obj.initial_params.delta = [];
                obj.initial_params.initial_state = [];

            catch exception
                error('An error occurred: %s.\n', exception.message)
            end
            
            DEBUG(obj.Q)
            DEBUG(obj.A)

        end

        function Pa = Pa_init(obj, varargin)
            p = inputParser;
            p.KeepUnmatched = true;

            DEBUG(varargin)

            default_delta = 1e-1;
            check_delta = @(x) isnumeric(x) && isscalar(x) && (x>0);
            addOptional(p, 'delta', default_delta, check_delta);
            
            % Insert new parameters to init the Pa here

            parse(p, varargin{:});
            try
                Pa = p.Results.delta * eye(obj.dim);
                obj.initial_params.delta = p.Results.delta;
            catch exception
                error('An error occurred: %s.\n', exception.message);
            end    
        end
        
        function xa = xa_init(obj, varargin)
            p = inputParser;
            p.KeepUnmatched = true;

            default_initial_state = zeros(obj.dim, 1);
            check_initial_state = @(x) isnumeric(x) && (length(x)==obj.dim);
            addOptional(p, 'initial_state', default_initial_state, check_initial_state);
            
            parse(p, varargin{:});
            try
                % if ~any(strcmp('initial_state', p.UsingDefaults)) && (length(p.Results.initial_state) ~= obj.dim)
                %     error('The initial state used is different than model dimension.')
                % end
                xa = p.Results.initial_state;
                obj.initial_params.initial_state = p.Results.initial_state;
            catch exception
                error('An error occurred: %s.\n', exception.message);
            end
        end

        function Q = update_Q(obj, varargin)
            DEBUG(varargin)
            Q = obj.Q;
        end
        
        function A = update_A(obj, varargin)
            DEBUG(varargin)
            A = obj.A;
        end

        function A = get_A_matrix(obj)
            A = obj.A;
        end
    end
end