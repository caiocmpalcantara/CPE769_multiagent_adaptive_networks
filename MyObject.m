classdef MyObject < handle
    properties (SetAccess = private)
        ID; % Unique identifier for the object
    end
    
    methods
        function obj = MyObject()
            persistent uniqueID; % Persistent variable to maintain state across all instances
            
            if isempty(uniqueID)
                uniqueID = 0; % Initialize the unique ID counter
            end
            
            uniqueID = uniqueID + 1; % Increment the unique ID for each new instance
            obj.ID = uniqueID; % Assign the incremented value to the object's ID property
        end
        function id = getID(obj)
            id = obj.ID;
        end
    end
end