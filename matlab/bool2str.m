function string = bool2str(b)

string = regexprep(sprintf('%i',boolean(b)),{'1','0'},{'True','False'});

end