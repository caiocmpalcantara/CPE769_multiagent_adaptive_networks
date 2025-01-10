classdef Kalman < Agent_technique
    properties (Access = public)
        state_model
        last_update_time
        Pp
        Pa
        Sigma_y
        xp_hat
        xa_hat
        VAR_R
        VAR_phi
        last_tx_interval
        % Statistics
        S
        K
        iteracts            % número de iterações 
    end
    methods (Abstract)
        y = update_obs(measure);
        y_hat = update_y_hat(obj);
        Py = update_Py(obj);
        Pxy = update_Pxy(obj);
        % Px = update_Px(obj, time);
        Sigma_y = update_Sigma_y(obj);
    end
    methods
        function obj = Kalman(pos0, pos1, t0, t1, range_sd, bearing_sd)
            
            % obj = Kalman(pos0, pos1, t0, t1, range_sd, bearing_sd)
            
            obj.state_model = Const_Speed(0.2);
            obj.last_update_time = t1;
            obj.last_tx_interval = seconds(t1-t0);
            obj.Pa = obj.state_model.Pa_init(pos0, pos1, range_sd, bearing_sd, obj.last_tx_interval);
            obj.Pp = obj.Pa;
            obj.Sigma_y = eye(2);
            obj.xa_hat = obj.state_model.xa_init(pos0, pos1, obj.last_tx_interval);
            obj.xp_hat = obj.xa_hat;
            obj.VAR_R = range_sd.^2;
            obj.VAR_phi = bearing_sd.^2;
            
            obj.S = zeros(2,2);
            obj.K = zeros(4,2);
            % obj.a = [0;0];
            obj.iteracts = 0;
        end
        function xa = prior_estimate_position(obj, time)
            if isa(obj.state_model, "State_Model")
                obj.state_model.update_A(time);
                x = obj.state_model.A * obj.xp_hat;
                % xa = xa(1:obj.state_model.get_model_order(), 1); %TODO: Necessitará refactoring para encapsular tipo de state_model; funciona apenas para CV
                xa = Displacement(x(1,1), x(2,1), 'Cartesian');
            else
                error('Unexpected object type.')
            end
        end
        function va = prior_estimate_speed(obj, time)
            if isa(obj.state_model, "State_Model")
                obj.state_model.update_A(time)
                v = obj.state_model.A * obj.xp_hat;
                va = Speed(v(3,1), v(4,1), 'Cartesian');
            else
                error('Unexpected object type.')
            end
        end
        function xp = posterior_estimate_position(obj)
            xp = Displacement(obj.xp_hat(1,1), obj.xp_hat(2,1), 'Cartesian');
        end
        function vp = posterior_estimate_speed(obj)
            vp = Speed(obj.xp_hat(3,1), obj.xp_hat(4,1), 'Cartesian'); 
        end
        
        function A = update_A(obj, time)
            obj.last_tx_interval = seconds(time - obj.last_update_time);
            A = obj.state_model.update_A(obj.last_tx_interval);
        end
        
        function Q = update_Sigma_x(obj, time)
            Q = obj.state_model.update_Sigma_x(seconds(time - obj.last_update_time));
        end
        
        function Px = update_Px(obj, time)
            t = seconds(time - obj.last_update_time);
            Px = obj.state_model.update_A(t) * obj.Pp * (obj.state_model.update_A(t))';
        end

        function y_hat = apply(obj, obs_buffer, state_buffer)
            obj.iteracts = obj.iteracts + 1;
            obj.update(obs_buffer(:,1), obj.last_update_time+obj.last_tx_interval);
            y_hat = obj.update_y_hat();
        end

        function update(obj, measure, time)          
            % Prior
            A = obj.update_A(time);
            obj.xa_hat = A * obj.xp_hat;
            obj.Pa = obj.update_Px(time) + obj.update_Sigma_x(time);

            % Update observation
            y = obj.update_obs(measure);

            % Posterior
            S = obj.update_Py() + obj.update_Sigma_y();
            K = obj.update_Pxy() / S;
            obj.Pp = obj.Pa - K * S * K';
            a = y - obj.update_y_hat();
            obj.xp_hat = obj.xa_hat + K * a;

            obj.last_update_time = time;

            % get statistics
            obj.S = S;
            obj.K = K;
            % obj.a = a;
        end

        function xa = get_prior_state(obj)
            xa = obj.xa_hat;
        end

        function xp = get_posterior_state(obj)
            xp = obj.xp_hat;
        end

        function set_priors(obj, varargin)
            p = inputParser;

            default_sig_R = 15;
            valid_sig_R = [1 1e2];
            check_sig_R = @(x) isnumeric(x) && all(and(x >= min(valid_sig_R), x <= max(valid_sig_R)));

            default_sig_phi = 2;
            valid_sig_phi = [.1 2e1];
            check_sig_phi = @(x) isnumeric(x) && all(and(x >= min(valid_sig_phi), x <= max(valid_sig_phi)));

            addOptional(p, 'sig_R', default_sig_R, check_sig_R);
            addOptional(p, 'sig_phi', default_sig_phi, check_sig_phi);

            parse(p, varargin{:});

            obj.VAR_R = (p.Results.sig_R).^2;
            obj.VAR_phi = (deg2rad(p.Results.sig_phi)).^2;
        end
    end
    methods (Static)
        function assign(kf1, kf2)
            try
                kf1.last_update_time = kf2.last_update_time;
                kf1.Pp = kf2.Pp;
                kf1.Pa = kf2.Pa;
                if(isa(kf1, 'PDNKF') && isa(kf2, 'PDNKF'))
                    kf1.Sigma_y = kf2.Sigma_y;
                end
                kf1.xp_hat = kf2.xp_hat;
                kf1.xa_hat = kf2.xa_hat;
                kf1.last_tx_interval = kf2.last_tx_interval;
            catch exception
                disp(exception);
            end 
        end
    end
end