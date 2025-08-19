classdef System_Model < handle
    %SYSTEM_MODEL: An abtract class for designing Kalman System Model
    %   The context based System Model is implemented in the derived classes.
    %   The Kalman State Model is context based. TODO: Need to think a more general methods in this adaptive filters.
    %
    %AGO2025 IPqM-GSAS, Alcantara.
    properties
        dim
    end
    methods (Abstract)
        Pa_init(obj, varargin);
        xa_init(obj, varargin);
        update_Q(obj, varargin);
        update_A(obj, varargin);
    end
    methods
        function obj = System_Model(varargin)
            p = inputParser;
            p.KeepUnmatched = true;

            % Defining mandatory arguments


            % Defining optional arguments
            default_dim = 2;
            check_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0);
            addOptional(p, 'dim', default_dim, check_dim);
            
            % DEBUG(varargin)
            % ind = find(strcmp(varargin, 'dim'));
            % if ~isempty(ind)
            %     DEBUG(varargin{ind+1})                
            % end
            try
                parse(p, varargin{:});    
                obj.dim = p.Results.dim;
                % DEBUG(obj.dim);                

            catch exception
                error("An error occurred: %s.\n", exception.message)
            end

            DEBUG(obj.dim)

        end

    end
    methods (Access = public)
        function model_order = get_model_order(obj)
            model_order = obj.dim;
        end
    end
end