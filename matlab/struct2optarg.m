function optional_arguments = struct2optarg(parameter_struct)
    
    fields = fieldnames(parameter_struct);
    optional_arguments = cell(1,numel(fields)*2);
    for i = 1:numel(fields)
        optional_arguments{1,2*i-1} = fields{i};
        optional_arguments{1,2*i} = parameter_struct.(fields{i});
    end

end