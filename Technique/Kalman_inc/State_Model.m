classdef State_Model < handle
    %STATE_MODEL: An abtract class for designing Kalman System Model
    %   The context based System Model is implemented in the derived classes.
    %   The Kalman State Model is context based. TODO: Need to think a more general methods in this adaptive filters.
    %
    %JAN2024 IPqM-GSAS, Alcantara.
    properties
        Sigma_x
        A
    end
    methods (Abstract)
        Pa_init(obj, pos0, pos1, range_sd, bearing_sd, tx_interval);
        xa_init(obj, pos0, pos1, tx_interval);
        update_Sigma_x(obj, tx_interval);
        update_A(obj, tx_interval);
    end
    methods (Access = public)
        function model_order = get_model_order(obj)
            model_order = size(obj.A, 1)/2;
        end
    end
end