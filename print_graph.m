% Define nodes and edges
s = [1 1 1 2 3 3 4 4 4 5 6 ];  % Source nodes
t = [1 2 3 2 3 4 4 5 6 5 6];  % Target nodes

% Create a graph object
G = graph(s, t);

% Plot the graph
plot(G);