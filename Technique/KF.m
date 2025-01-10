classdef KF < Kalman
    properties
        H
    end

    methods
        function obj = KF(pos0, pos1, t0, t1, range_sd, bearing_sd)
            % obj = Kalman(pos0, pos1, t0, t1, range_sd, bearing_sd);
            obj@Kalman(pos0, pos1, t0, t1, range_sd, bearing_sd);
            N = obj.state_model.get_model_order();
            
            obj.H = zeros(N, 2*N);
            obj.H(1,1) = 1;
            obj.H(2,2) = 1;
            
            obj.Sigma_y(1,1) = obj.VAR_R;
            obj.Sigma_y(2,2) = obj.VAR_R;
        end
        function y = update_obs(obj, measure)
            y = zeros(2,1);
            y(1,1) = measure.get_cart(1);
            y(2,1) = measure.get_cart(2);
        end
        function y_hat = update_y_hat(obj)
            y_hat = obj.H * obj.xa_hat;
        end
        function Py = update_Py(obj)
            Py = obj.H * obj.Pa * obj.H';
        end
        function Pxy = update_Pxy(obj)
            Pxy = obj.Pa * obj.H';
        end
        function Px = update_Px(obj, time)
            A = obj.update_A(time);
            % A = update_A@Kalman(obj, time);
            Px = A * obj.Pp * A';
        end
        function Sigma_y = update_Sigma_y(obj)
            Sigma_y = obj.Sigma_y;
            Sigma_y(1,1) = obj.VAR_R;
            Sigma_y(2,2) = obj.VAR_R;
        end
        function H = get_H(obj)
            H = obj.H;
        end
        function obj = update_H(obj, H_new)
            obj.H = H_new;
        end
    end
end