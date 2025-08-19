classdef Sensor_Model < handle
    %SENSOR_MODEL: An abtract class for designing Kalman System Model
    %   The context based System Model is implemented in the derived classes.
    %   The Kalman Sensor Model is context based. TODO: Need to think a more general methods in this adaptive filters.
    %
    %AGO2025 IPqM-GSAS, Alcantara.
    properties
        dim
    end
    methods (Abstract)
        % Pa_init(obj, varargin);
        % xa_init(obj, varargin);
        update_R(obj, varargin);
        update_H(obj, varargin);
    end
    methods
        function obj = Sensor_Model(varargin)
            p = inputParser;

            % Defining mandatory arguments
            check_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0);
            addRequired(p, 'dim', check_dim);
            
            try
                parse(p, varargin{:});    
                obj.dim = p.Results.dim;                

            catch exception
                error("An error occurred: %s.\n", exception.message)
            end
        end

    end
    methods (Access = public)
        function model_order = get_model_order(obj)
            model_order = obj.dim;
        end
    end
end