classdef GeoPoint < handle
    properties (Constant)
        R = 6400e3;   %[m]
    end
    properties (Access = protected)
        lat
        long
    end
    methods
        function obj = GeoPoint(var1, var2, varargin)
            %TODO: defensive programming and robust parameters
            if nargin == 2
                obj.lat = var1;
                obj.long = var2;
            elseif nargin == 4
                switch varargin{1}
                    case 'Polar'
                        r = var1;
                        phi = var2;
                        x = r*cos(phi);
                        y = r*sin(phi);
                        obj = GeoPoint(x, y, 'Cartesian', varargin{2});
                    case 'Cartesian'
                        %TODO: more realistic x y convertion
                        x = var1;
                        y = var2;
                        GeoPointRef = varargin{2};

                        R = 6400e3;     %[m]
                        theta = y/R;    %[rad]
                        lambda = x/(R*cos(theta));  %[rad]
                        try
                            %TODO: lat long condições de contorno (próximos aos polos, próximo a transição de 180 para -180 de longitude)
                            lat = GeoPointRef.get_lat() + rad2deg(theta);
                            long = GeoPointRef.get_long() + rad2deg(lambda);
                            obj.lat = lat;
                            obj.long = long;
                        catch exception
                            disp(exception)
                        end
                    otherwise
                        error('Error: wrong input format: ´Polar´ or ´Cartesian´')
                end
            else
                error('Error: wrong input format: nargin')
            end
        end
        function lat = get_lat(obj)
            lat = obj.lat;
        end
        function long = get_long(obj)
            long = obj.long;
        end
        function result = minus(obj1, obj2) % for short differences 
            if(isa(obj1, 'GeoPoint') && isa(obj2, 'GeoPoint'))
                d_lat = deg2rad(obj1.get_lat() - obj2.get_lat());
                d_long = deg2rad(obj1.get_long() - obj2.get_long());
                dx = obj1.R * d_long;
                dy = obj1.R * d_lat;
                r = norm([dx dy], 2);
                phi = atan2(dy, dx);
                result = Displacement(r, phi, 'Polar');
            elseif(isa(obj1, 'GeoPoint') && isa(obj2, 'Displacement'))
                d = Displacement(-obj2.get_cart(1), -obj2.get_cart(2), 'Cartesian');
                result = obj1 + d;
            end
        end
        function result = plus(obj1, obj2)
            if (isa(obj1, 'Displacement') && isa(obj2, 'GeoPoint'))
                % result = GeoPoint(obj1.x.get_m(), obj1.y.get_m(), 'Cartesian', obj2);
                result = GeoPoint(obj1.get_cart(1), obj1.get_cart(2), 'Cartesian', obj2);
            elseif (isa(obj1, 'GeoPoint') && isa(obj2, 'Displacement'))
                % result = GeoPoint(obj2.x.get_m(), obj2.y.get_m(), 'Cartesian', obj1);
                result = GeoPoint(obj2.get_cart(1), obj2.get_cart(2), 'Cartesian', obj1);
            else
                error('The arguments must to be Displacement and GeoPoint.')
            end
        end
    end
end