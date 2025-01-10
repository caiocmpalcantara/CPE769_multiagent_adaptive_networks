classdef Speed < handle
    properties
        vx
        vy
        vr
        phi
    end

    methods
        function obj = Speed(var1, var2, varargin) % TODO: More robust constructor
            if nargin == 2
                if isa(var1, 'Displacement') && isnumeric(var2)
                    obj.vx = var1.get_cart(1)/var2;
                    obj.vy = var1.get_cart(2)/var2;
                    obj.vr = var1.get_polar(1)/var2;
                    obj.phi = var1.get_polar(2);
                else
                    error('The first input is not from Displacement Class.')
                end
            elseif nargin == 3
                if isnumeric(var1) && isnumeric(var2)
                    switch varargin{1}
                        case 'Cartesian'
                            obj.vx = var1;
                            obj.vy = var2;
                            obj.vr = norm([var1 var2], 2);
                            obj.phi = atan2(var2, var1);
                        case 'Polar'
                            obj.vx = var1*cos(var2);
                            obj.vy = var1*sin(var2);
                            obj.vr = var1;
                            obj.phi = wrapToPi(var2);
                        otherwise
                            error('Wrong type of symmetry: `Cartesian` or `Polar`.')
                    end
                else
                    error('Wrong input type: the type is not numeric.')
                end
            else
                error('Wrong number of inputs.')
            end
        end
        function result = times(obj, t)
            if isnumeric(t)
                result = Displacement(obj.vx*t, obj.vy*t, 'Cartesian');
            end
        end
        function result = get_cart(obj, varargin) %TODO: More robust getter
            if nargin == 2
                switch varargin{1}
                    case 1
                        result = [obj.vx];
                    case 2
                        result = [obj.vy];
                    otherwise
                        result = [obj.vx; obj.vy];      %[m/s]
                end
            else
                result = [obj.vx; obj.vy];      %[m/s]
            end
        end
        function result = get_polar(obj, varargin) %TODO: More robust getter
            if nargin == 2
                switch varargin{1}
                    case 1
                        result = obj.vr;
                    case 2
                        result = obj.phi;
                    otherwise
                        result = [obj.vr obj.phi]';        
                end
            else
                result = [obj.vr; obj.phi];     %[m/s] and [rad]
            end
            
        end
    end
    methods (Static)
        function array = array(dims)
            N = prod(dims);
            for n = 1:N
                array(n) = Speed(0,0,'Cartesian');
            end
            array = reshape(array, dims);
        end
    end
end