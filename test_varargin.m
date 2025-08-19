function output = test_varargin(varargin)
    cell_list = {1, 2, 'delta', 3};
    output = [isa(varargin, 'cell') isa(varargin, "double") isa(cell_list, 'cell')];
end