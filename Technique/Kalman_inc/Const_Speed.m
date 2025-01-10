classdef Const_Speed < State_Model
    properties (Access = protected)
        VAR_speed
    end

    methods
        function obj = Const_Speed(speed_sd)
            obj.VAR_speed = speed_sd.^2;

            obj.Sigma_x = zeros(4,4);
            obj.Sigma_x(3,3) = obj.VAR_speed;
            obj.Sigma_x(4,4) = obj.VAR_speed;

            obj.A = eye(4);
        end
        function Pa = Pa_init(obj, pos0, pos1, range_sd, bearing_sd, tx_interval)
            try
                rx0 = pos0.get_cart(1);
                ry0 = pos0.get_cart(2);
                rx1 = pos1.get_cart(1);
                ry1 = pos1.get_cart(2);
                phi0 = atan2(ry0, rx0);
                phi1 = atan2(ry1, rx1);
                dt = tx_interval;

                Pa = eye(4);
                Pa(1,1) = ry1^2*bearing_sd^2 + range_sd^2*cos(phi1)^2*(1-2*bearing_sd^2) + ...
                                                (range_sd*bearing_sd)^2 + (rx1 * bearing_sd^2/2)^2;
                Pa(1,2) = range_sd^2*(1-2*bearing_sd^2)*sin(2*phi1)/2 - ...
                            rx1*ry1*bearing_sd^2 + rx1 * ry1 * bearing_sd^4/4;
                Pa(1,3) = Pa(1,1)/dt - rx1 * rx0 * bearing_sd^4/4/dt;
                Pa(1,4) = Pa(1,2)/dt - rx1 * ry0 * bearing_sd^4/4/dt;
                Pa(2,1) = Pa(1,2);
                Pa(2,2) = rx1^2*bearing_sd^2 + range_sd^2*sin(phi1)^2*(1-2*bearing_sd^2) + ...
                                                range_sd^2*bearing_sd^2 + (ry1 * bearing_sd^2/2)^2;
                Pa(2,3) = Pa(1,2)/dt - ry1 * rx0 * bearing_sd^4/4/dt;
                Pa(2,4) = Pa(2,2)/dt - ry1 * ry0 * bearing_sd^4/4/dt;
                Pa(3,1) = Pa(1,3);
                Pa(3,2) = Pa(2,3);
                Pa(3,3) = Pa(1,1) + ry0^2*bearing_sd^2 + range_sd^2*cos(phi0)^2*(1-2*bearing_sd^2) + ...
                                        range_sd^2*bearing_sd^2 + (rx0^2 - 2*rx1 * rx0) * bearing_sd^4/4;
                Pa(3,3) = Pa(3,3)/dt^2;
                Pa(3,4) = Pa(1,2) + range_sd^2*(1-2*bearing_sd^2)*sin(2*phi0)/2 - rx0*ry0*bearing_sd^2 + ...
                                        (rx0*ry0 - rx1*ry0 - ry1*rx0) * bearing_sd^4/4;
                Pa(3,4) = Pa(3,4)/dt^2;
                Pa(4,1) = Pa(1,4);   
                Pa(4,2) = Pa(2,4);
                Pa(4,3) = Pa(3,4);
                Pa(4,4) = Pa(2,2) + rx0^2*bearing_sd^2 + range_sd^2*sin(phi0)^2*(1-2*bearing_sd^2) + ...
                                        range_sd^2*bearing_sd^2 + (ry0^2 - 2*ry1 * ry0) * bearing_sd^4/4;
                Pa(4,4) = Pa(4,4)/dt^2;
            catch exception
                error(exception);
            end
            
        end
        function xa = xa_init(obj, pos0, pos1, tx_interval)
            try
                rx0 = pos0.get_cart(1);
                ry0 = pos0.get_cart(2);
                rx1 = pos1.get_cart(1);
                ry1 = pos1.get_cart(2);
                dt = tx_interval;

                xa = zeros(4,1);
                xa(:,1) = [ (rx0+rx1)/2; ...
                            (ry0+ry1)/2; ...
                            (rx1-rx0)/(dt*3); ...
                            (ry1-ry0)/(dt*3)];
            catch exception
                error(exception);
            end
        end
        function Sigma_x = update_Sigma_x(obj, tx_interval)
            Sigma_x = obj.Sigma_x;
        end
        function A = update_A(obj, tx_interval)
            obj.A(1,3) = tx_interval;
            obj.A(2,4) = tx_interval;
            A = obj.A;
        end
    end
end