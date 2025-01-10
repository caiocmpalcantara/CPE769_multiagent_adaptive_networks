classdef Displacement < handle
    properties (Access = protected)
        x
        y
        r
        phi
    end

    methods
        function obj = Displacement(var1, var2, type)
            %TODO: More robust contructor
            switch type
                case 'Cartesian'
                    obj.x = var1;   % in [m]
                    obj.y = var2;   % in [m]
                    obj.r = norm([var1 var2], 2);
                    obj.phi = atan2(var2, var1);
                case 'Polar'
                    obj.x = var1*cos(var2);
                    obj.y = var1*sin(var2);
                    obj.r = var1;   % in [m]
                    obj.phi = wrapToPi(var2); % in [rad]
                otherwise
                    error('Wrong type of input. It must to be `Cartesian` or `Polar`.')
            end
        end
        function result = get_cart(obj, varargin) %TODO: More robust getter
            if nargin == 2
                switch varargin{1}
                    case 1
                        result = [obj.x];   %Permite formação de vetores ao invés de um único número
                    case 2
                        result = [obj.y];
                    otherwise
                        result = [obj.x obj.y]';        
                end
            else
                result = [obj.x obj.y]';    %[m]
            end
        end
        function result = get_polar(obj, varargin) %TODO: More robust getter
            if nargin == 2
                switch varargin{1}
                    case 1
                        result = [obj.r];
                    case 2
                        result = [obj.phi];
                    otherwise
                        result = [obj.r obj.phi]';        
                end
            else
                result = [obj.r obj.phi]';  %[m] and [rad]
            end
        end
        function set_cart(obj, x, y) %TODO: More robust setter
            obj.x = x;  % [m]
            obj.y = y;  % [m]
            obj.r = norm([x y], 2);
            obj.phi = atan2(y,x);
        end
        function set_polar(obj, r, phi) %TODO: More robust setter
            obj.r = r;      % [m]
            obj.phi = phi;  % [rad]
            obj.x = r*cos(phi);
            obj.y = r*sin(phi);
        end
        function result = plus(obj1, obj2)
            if(isa(obj1, 'Displacement') && isa(obj2, 'Displacement'))
                result = Displacement(obj1.get_cart(1)+obj2.get_cart(1), obj1.get_cart(2)+obj2.get_cart(2), 'Cartesian');
            end
        end
        function result = minus(obj1, obj2)
            if(isa(obj1, 'Displacement') && isa(obj2, 'Displacement'))
                result = Displacement(obj1.get_cart(1)-obj2.get_cart(1), obj1.get_cart(2)-obj2.get_cart(2), 'Cartesian');
            end 
        end
        function result = rdivide(obj, t)
            if isnumeric(t)
                result = Speed(obj, t); %[m/s]
            end
        end
        function result = copy(obj)
            result = Displacement(obj.get_cart(1), obj.get_cart(2), 'Cartesian');
        end
    end
    methods (Static)
        function array = array(dims)
            N = prod(dims);
            for n = 1:N
                array(n) = Displacement(0,0,'Cartesian');
            end
            array = reshape(array, dims);
        end
    end
end