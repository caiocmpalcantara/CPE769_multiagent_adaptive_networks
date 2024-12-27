classdef Agent_technique < handle
    properties
        x_dim               % dimensão do estado
        y_dim               % dimensão da observação
    end
    methods (Abstract)
        
    end
    
    methods 
        function obj = Agent_technique(varargin)
            p = inputParser;

            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);

            default_y_dim = 1;
            check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);

            addOptional(p, 'x_dim', default_x_dim, check_x_dim);
            addOptional(p, 'y_dim', default_y_dim, check_y_dim);

            parse(p, varargin{:});

            try
                obj.x_dim = p.Results.x_dim;
                obj.y_dim = p.Results.y_dim;
            catch exception
                error('An error occurred %s', exception.message);
            end

        end
        % function n_win = get_win_length(obj)
        %     n_win = obj.n_win;
        % end
    end

end