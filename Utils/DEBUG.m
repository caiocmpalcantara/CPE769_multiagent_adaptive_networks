function DEBUG(obj)
    % Retrieve caller information
    st = dbstack('-completenames', 1);
    if numel(st) >= 1
        caller = st(1);
        fname = caller.file;
        funcName = caller.name;
        lineNum = caller.line;
    else
        fname = '<unknown>';
        funcName = '<command-line>';
        lineNum = NaN;
    end
    
    % Print debug header
    fprintf('[DEBUG] Called from file: %s\n', fname);
    fprintf('[DEBUG] In function: %s, at line: %d\n', funcName, lineNum);

    % Display object/value and its class
    try
        disp(obj);
    catch
        fprintf('Could not display object directly.\n');
    end
    fprintf('Type/Class: %s\n\n', class(obj));
end
