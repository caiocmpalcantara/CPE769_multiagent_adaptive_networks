classdef Kalman_ < Agent_technique
    properties (Access = public)
        system_model        % The Kalman system model (state dynamics)
        sensor_model        % The Kalman sensor model (measurement dynamics: the observation)
        last_update_time    % The last timestamp that occurred a detection
        xp_hat              % Posterior state estimate
        xa_hat              % Prior state estimate
        last_T_sample       % Time passed between samples 
        % Statistics
        Pp                  % Posterior state error covariance matrix
        Pa                  % Prior state error covariance matrix
        S                   % Innovation covariance matrix
        K                   % Kalman gain matrix
        iteracts            % número de iterações 
    end
    methods (Abstract)
        y = update_obs(measure);
        y_hat = update_y_hat(obj);
        Py = update_Py(obj);
        Pxy = update_Pxy(obj);
        % Px = update_Px(obj, time);
        R = update_R(obj);
    end
    methods
        function obj = Kalman_(varargin)
            % obj = Kalman(pos0, pos1, t0, t1, range_sd, bearing_sd)
            
            p = inputParser;
            
            % Optional arguments
            default_system = Linear_State();
            check_system = @(x) isa(x, "State_Model");
            addOptional(p, 'system_model', default_system, check_system)
            % obj.system_model = Const_Speed(0.2);
            
            default_T_sample = 1;
            check_T_sample = @(x) isnumeric(x) && isscalar(x) && (x>0);
            addOptional(p, 'T', default_T_sample, check_T_sample);
            
            default_last_update_time = datetime("now");
            check_last_update_time = @(x) isa(x, 'datetime');
            addOptional(p, 'timestamp', default_last_update_time, check_last_update_time);
    
            % obj.last_update_time = t1;
            % obj.last_T_sample = seconds(t1-t0);
            
            default_sensor_model = Linear_Obs();
            check_sensor_model = @(x) isa(x, "Sensor_Model");
            addOptional(p, 'sensor_model', default_sensor_model, check_sensor_model);

            try
                parse(p, varargin{:});
                
                % System
                obj.system_model = p.Results.system_model;
                obj.Pa = obj.system_model.Pa_init(varargin);        % Verificar questoes de conflito no parser
                obj.Pp = obj.Pa;
                obj.xa_hat = obj.system_model.xa_init(varargin);    % Verificar questoes de conflito no parser
                obj.xp_hat = obj.xp_hat;

                % Sensor
                obj.sensor_model = p.Results.sensor_model;

                % Kalman
                obj.K = zeros(obj.system_model.dim, obj.sensor_model.dim);
                obj.S = zeros(obj.sensor_model.dim);

            catch exception
                
            end


            obj.Pa = obj.system_model.Pa_init(pos0, pos1, range_sd, bearing_sd, obj.last_T_sample);
            obj.Pp = obj.Pa;
            obj.R = eye(2);
            obj.xa_hat = obj.system_model.xa_init(pos0, pos1, obj.last_T_sample);
            obj.xp_hat = obj.xa_hat;
            o = range_sdj.VAR_phi = bearing_sd.^2;
            
            obj.S = zeros(2,2);
            obj.K = zeros(4,2);
            % obj.a = [0;0];
            obj.iteracts = 0;
        end
        function xa = prior_estimate_position(obj, time)
            if isa(obj.system_model, "system_Model")
                obj.system_model.update_A(time);
                x = obj.system_model.A * obj.xp_hat;
                % xa = xa(1:obj.system_model.get_model_order(), 1); %TODO: Necessitará refactoring para encapsular tipo de system_model; funciona apenas para CV
                xa = Displacement(x(1,1), x(2,1), 'Cartesian');
            else
                error('Unexpected object type.')
            end
        end
        function va = prior_estimate_speed(obj, time)
            if isa(obj.system_model, "system_Model")
                obj.system_model.update_A(time)
                v = obj.system_model.A * obj.xp_hat;
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
            obj.last_T_sample = seconds(time - obj.last_update_time);
            A = obj.system_model.update_A(obj.last_T_sample);
        end
        
        function Q = update_Sigma_x(obj, time)
            Q = obj.system_model.update_Sigma_x(seconds(time - obj.last_update_time));
        end
        
        function Px = update_Px(obj, time)
            t = seconds(time - obj.last_update_time);
            Px = obj.system_model.update_A(t) * obj.Pp * (obj.system_model.update_A(t))';
        end

        function y_hat = apply(obj, obs_buffer, state_buffer)
            obj.iteracts = obj.iteracts + 1;
            obj.update(obs_buffer(:,1), obj.last_update_time+obj.last_T_sample);
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
            S = obj.update_Py() + obj.update_R();
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

            o = (p.Results.sig_R)j.VAR_phi = (deg2rad(p.Results.sig_phi)).^2;
        end
    end
    methods (Static)
        function assign(kf1, kf2)
            try
                kf1.last_update_time = kf2.last_update_time;
                kf1.Pp = kf2.Pp;
                kf1.Pa = kf2.Pa;
                if(isa(kf1, 'PDNKF') && isa(kf2, 'PDNKF'))
                    kf1.R = kf2.R;
                end
                kf1.xp_hat = kf2.xp_hat;
                kf1.xa_hat = kf2.xa_hat;
                kf1.last_T_sample = kf2.last_T_sample;
            catch exception
                disp(exception);
            end 
        end
    end
end