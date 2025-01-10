classdef Measure < handle
    properties
    end
    methods (Abstract)
        % diplacement
        measure = get_m(obj)
        measure = get_mn(obj)
        measure = get_km(obj)
        measure = get_ky(obj)
        % speed
        measure = get_m_s(obj)
        measure = get_km_h(obj)
        measure = get_knot(obj)
    end
end