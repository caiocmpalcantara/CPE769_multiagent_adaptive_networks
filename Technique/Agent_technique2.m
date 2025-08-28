classdef Agent_technique2 < handle
    properties
        x_dim               % dimensão do estado
        y_dim               % dimensão da observação
    end
    methods (Abstract)
        [varargout] = apply(obj, varargin)    % varargin -> (Wiener) obs_buffder, and state_buffer; (Kalman) current obs and other Kalman options
        obj = reset(obj);
        obj = update_params(obj, varargin);
        obj = get_params(obj, varargin);
    end
    
    methods 
        function obj = Agent_technique2(varargin)
            p = inputParser;
            p.KeepUnmatched = true;

            % DEBUG(varargin)

            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);
            addOptional(p, 'x_dim', default_x_dim, check_x_dim);

            default_y_dim = 1;
            check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);
            addOptional(p, 'y_dim', default_y_dim, check_y_dim);

            parse(p, varargin{:});

            try
                obj.x_dim = p.Results.x_dim;
                obj.y_dim = p.Results.y_dim;
            catch exception
                error('An error occurred %s', exception.message);
            end

            DEBUG(obj.x_dim)
            DEBUG(obj.y_dim)
            
        end
        % function n_win = get_win_length(obj)
        %     n_win = obj.n_win;
        % end
    end

    methods (Static)
        % Expectation de matrizes e vetores, de acordo com a dimensão dos buffers
        function out = expectation_approx(buff1, buff2, N)
            % N amostras temporais
            out = zeros(size(buff1,1), size(buff2,1));
            for i = 1:N
                out = buff1(:,i) * buff2(:,i)' + out;
            end
            out = out / N;
        end
    end

end