neighbor_lists = {
    [1, 3];      % Agent 1 neighbors (including self)
    [1, 2];         % Agent 2 neighbors
    [1, 3, 4];      % Agent 3 neighbors
    [3, 4, 5, 6];   % Agent 4 neighbors
    [4, 5];         % Agent 5 neighbors
    [4, 5, 7];         % Agent 6 neighbors
    [7, 8];         % Agent 7 neighbors
    [7, 8];         % Agent 8 neighbors
};

[s,t] = print_graph(neighbor_lists);