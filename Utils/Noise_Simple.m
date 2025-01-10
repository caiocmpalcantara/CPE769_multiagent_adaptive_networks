classdef Noise_Simple < Noise_Model
    properties (Access = protected)
        a
        b
        signal_param
    end
    methods
        function obj = Noise_Simple(varargin)
            if nargin == 0
                obj.signal_param = 17e3;
                obj.a = 1e-6;
                obj.b = 10;
            elseif nargin == 1
                obj.signal_param = varargin{1};
                obj.a = 1e-6;
                obj.b = 10;
            elseif nargin == 2
                obj.signal_param = varargin{1};
                obj.a = varargin{2};
                obj.b = 10;
            elseif nargin == 3
                obj.signal_param = varargin{1};
                obj.a = varargin{2};
                obj.b = varargin{3};
            else
                error('Wrong number of inputs.');
            end
        end
        function value = rand(obj, signal_param, varargin)
            if isa(signal_param, 'WindowType')
                STD = 0; %TODO: More complex model based on AF
            elseif isa(signal_param, 'Waveform')
                STD = 0; %TODO: More complex model based on AF
            else
                STD = (obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b; %TODO: DO A GENERAL FUNCTION AS A OBJ PARAM TO AVOID FUNCTION MISTAKES IN GETTERS
            end

            f = @(x, y) x*randn(y);
            
            points = 1;

            if nargin >= 3
                switch varargin{1}
                    case 'Gaussian'
                        f = @(x, y) x*randn(y);
                    case 'Uniform'
                        f = @(x, y) x*rand(y);
                    otherwise
                        f = @(x, y) x*randn(y);
                end
            end
            if nargin == 4
                points = varargin{2};
            elseif nargin > 4
                error('Wrong number of inputs.');
            end

            value = f(STD, points); 
        end
        function VAR = get_VAR(obj, signal_param, RV_type)
            switch RV_type
                case 'Gaussian'
                    VAR = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b)^2;
                case 'Uniform'
                    %TODO: Fix this
                    VAR = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b)^2;
                otherwise
                    VAR = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b)^2;
            end
        end
        function VAR = get_estimated_VAR(obj, signal_param, N, RV_type)
            if (isnumeric(N) && N > 0)
                realizations = obj.rand(signal_param, RV_type, [N 1]);
                VAR = var(realizations);
            else
                error('The number N of realizations must be a number and greater than 0.');
            end
        end
        function STD = get_STD(obj, signal_param, varargin)
            if nargin == 3
                RV_type = varargin{1};
            else
                RV_type = 'Gaussian';
            end

            switch RV_type
                case 'Gaussian'
                    STD = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b);
                case 'Uniform'
                    %TODO: Fix this
                    STD = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b);
                otherwise
                    STD = ((obj.a*abs(signal_param - obj.signal_param)).^2 + obj.b);
            end
        end
        function STD = get_estimated_STD(obj, signal_param, N, RV_type)
            if (isnumeric(N) && N > 0)
                realizations = obj.rand(signal_param, RV_type, [N 1]);
                STD = std(realizations);
            else
                error('The number N of realizations must be a number and greater than 0.');
            end
        end
        function MEAN = get_MEAN(obj, signal_param)
            MEAN = 0;
        end
        function MEAN = get_estimated_MEAN(obj, signal_param, N, RV_type)
            if (isnumeric(N) && N > 0)
                realizations = obj.rand(signal_param, RV_type, [N 1]);
                MEAN = mean(realizations);
            else
                error('The number N of realizations must be a number and greater than 0.');
            end
        end
        function [a b signal_param] = get_params(obj)
            a = obj.a;
            b = obj.b;
            signal_param = obj.signal_param;
        end
        function set_a(obj, a)
            obj.a = a;
        end
        function set_b(obj, b)
            obj.b = b;
        end
        function set_param(obj, param)
            obj.signal_param = param;            
        end
        function varargout = print_curve(obj, param_interval, N_points)
            d = param_interval(2)-param_interval(1);
            x = param_interval(1) : d/N_points : param_interval(2);
            y = (obj.a*abs(x - obj.signal_param)).^2 + obj.b;

            %subplot(2,1,1)
            plot(x,y,'-r','LineWidth', 1.5)
            ylabel('Desvio')
            xlabel('Parâmetro')
            grid on
            % subplot(2,1,2)
            % plot(x,y.^2,'-r','LineWidth', 1.5)
            % ylabel('Variância')
            % xlabel('Parâmetro')
            % grid on

            if nargout == 2
                varargout{1} = x;
                varargout{2} = y;
            end

        end
    end
end