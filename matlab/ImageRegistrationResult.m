classdef ImageRegistrationResult
    %IMAGEREGISTRATIONRESULT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        im1_registered
        im2_registered
        im1_points  % not implemented
        im2_points  % not implemented
        scaleRecovered
        thetaRecovered
        percentPtsUsed
        tform
        tform_status
        min_pts_used = true
        registration_successful = false
        match = true % default as failed match
        message = ''
    end
    
    methods
    end
    
end

