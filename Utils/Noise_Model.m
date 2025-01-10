classdef Noise_Model < handle
    methods (Abstract)
        value = rand(obj, signal, varargin);
        VAR = get_VAR(obj, signal_param);
        VAR = get_estimated_VAR(obj, signal_param, N);
        VAR = get_STD(obj, signal_param);
        VAR = get_estimated_STD(obj, signal_param, N);
        MEAN = get_MEAN(obj, signal_param);
        MEAN = get_estimated_MEAN(obj, signal_param, N);
        print_curve(obj, param_interval);
    end
end