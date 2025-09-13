function varargout = print_graph(neighbor_lists)
    % PRINT_GRAPH Visualizes a network graph from neighbor lists
    %
    % Input:
    %   neighbor_lists - Cell array where each cell contains neighbor indices
    %                   for that agent (including self-connections)
    %
    % Example:
    %   neighbor_lists = {[1,2,3]; [1,2]; [1,3,4]; [3,4,5,6]; [4,5]; [4,6]};
    %   print_graph(neighbor_lists);
    
    % Input validation
    if ~iscell(neighbor_lists)
        error('Input must be a cell array');
    end
    
    if isempty(neighbor_lists)
        error('Input cell array cannot be empty');
    end
    
    % Initialize source and target arrays
    s = [];
    t = [];
    
    % Convert neighbor lists to edge arrays
    for i = 1:length(neighbor_lists)
        neighbors = neighbor_lists{i};
        
        % Validate that neighbors is a numeric array
        if ~isnumeric(neighbors)
            error('Each cell must contain a numeric array of neighbor indices');
        end
        
        % Add edges from agent i to each of its neighbors
        for j = 1:length(neighbors)
            s = [s, i];
            t = [t, neighbors(j)];
        end
    end
    
    % Create edge lookup for fast bidirectional checking
    edge_set = containers.Map();
    for idx = 1:length(s)
        key = sprintf('%d_%d', s(idx), t(idx));
        edge_set(key) = true;
    end

    % Build filtered edge lists (removing duplicate bidirectional edges)
    s_final = [];
    t_final = [];
    edge_types = []; % 1=self, 2=bidirectional, 3=unidirectional

    for idx = 1:length(s)
        if s(idx) == t(idx)
            % Self-connection - always include
            s_final = [s_final, s(idx)];
            t_final = [t_final, t(idx)];
            edge_types = [edge_types, 1];
        else
            % Check if reverse edge exists
            reverse_key = sprintf('%d_%d', t(idx), s(idx));
            if isKey(edge_set, reverse_key)
                % Bidirectional - only include if s < t to avoid duplicates
                if s(idx) < t(idx)
                    s_final = [s_final, s(idx)];
                    t_final = [t_final, t(idx)];
                    edge_types = [edge_types, 2];
                end
            else
                % Unidirectional - include with reversed direction for information flow
                s_final = [s_final, t(idx)];  % Swap: target becomes source
                t_final = [t_final, s(idx)];  % Swap: source becomes target
                edge_types = [edge_types, 3];
            end
        end
    end

    % Create graph object with filtered edges
    G = digraph(s_final, t_final);

    % Plot the graph with customizations
    h = plot(G, 'NodeFontSize', 14, 'MarkerSize', 10, 'ArrowSize', 0);

    % Highlight different edge types based on filtered edges
    self_mask = (edge_types == 1);
    bidirectional_mask = (edge_types == 2);
    unidirectional_mask = (edge_types == 3);

    % Self-connections in red
    if any(self_mask)
        highlight(h, s_final(self_mask), t_final(self_mask), 'EdgeColor', 'red', 'LineWidth', 2);
    end

    % Bidirectional connections in blue (no arrows)
    if any(bidirectional_mask)
        highlight(h, s_final(bidirectional_mask), t_final(bidirectional_mask), 'EdgeColor', 'blue', 'LineWidth', 2, 'ArrowSize', 0);
    end

    % Unidirectional connections in green (with arrows showing information flow direction)
    if any(unidirectional_mask)
        highlight(h, s_final(unidirectional_mask), t_final(unidirectional_mask), 'EdgeColor', 'green', 'LineWidth', 2, 'ArrowSize', 13);
    end

    % Customize appearance
    title('Network Graph', 'FontSize', 16);
    xlabel('Node', 'FontSize', 12);
    ylabel('Node', 'FontSize', 12);

    if nargout == 1
        varargout{1} = s;
    elseif nargout == 2
        varargout{1} = s;
        varargout{2} = t;
    end
end
