classdef ImageRegistration < handle
% ImageRegistration Class
%   Class for processing various image registration methods
%
%   TODO: Input argument parameter documentation
    
    %% PROTECTED PARAMETERS
    properties %(Access = protected)
        
        % General Parameters & Flags
        method
        im1
        im1_fullres
        im2
        im2_fullres
        fixedPoints
        movingPoints
        visuals        
        matchedOriginal
        matchedDistorted
        images_unset
        
        % CNN Parameters & Flags
        cnn_window
        weights
        weights_col
        network
        network_col
        model
        model_col
        cnn_path
        model_loaded = false
        num_cnn_features
        
        % SeqReg Parameters & Flags
        step_size
        seq_len
        ps
        search_fract
        border
        num_steps
        strongest_match = 0
        random_startpoints = false
        semi_random_startpoints = false
        random_trajectories = false
        fullspread = false
        trajectory_mode
        random_trajectories_experimental = false
        random_scale
        traj_strengths = [];
        filter_points = false;
        fraction_pts_used
        min_pts_used
        min_pts_required = 4;
        min_fraction_pts_used = 0.1
        curr_seq_len
        curr_step_size
        curr_ps
        seqreg_showallmatches = true
        test_points_used = false
        
    end

    %% PUBLIC METHODS
    methods

        function obj = ImageRegistration(varargin)
            % ImageRegistration Constructor
            %   Object for various image registration methods
            %
            %   TODO: Input argument parameter documentation

            % parse input arguments
            obj.parseInput(varargin);
            
            % set flags
            obj.images_unset = true;
            
            obj.load_model();
            
        end

        function set_images(obj,im1f,im2f,im1s,im2s)
            % ImageRegistration.SET_IMAGES(im1f,im2f,im1s,im2s)
            %   Manually supply the full resolution and preprocessed 
            %   images to be used in image registration.
            %
            % Inputs
            %   im1f,im2f: Full resolution images
            %   
            %   im1s,im2s: Low resolution and preprocessed images for image
            %                   registration
        
            obj.im1 = im1s;
            obj.im2 = im2s;
            obj.im1_fullres = im1f;
            obj.im2_fullres = im2f;
            obj.images_unset = false;
        end
        
        function toggle_vis(obj)
        % ImageRegistration.TOGGLE_VIS()
        %   Toggle visualisations on//off

            obj.visuals = ~obj.visuals;
            
        end
        
        function set_trajectory_mode(obj,mode)
            % ImageRegistration.SET_TRAJECTORY_MODE(mode)
            %   Changes the trajectory mode to use for SeqReg method. 
            %
            % Inputs
            %   mode: an integer for the following options
            %   
            %       0: Uniform start points, uniform trajectories (ACRA16)
            %       1: Uniform start points, random trajectories 
            %       2: Full spread uniform start, random trajectories
            %               (ICRA17)
            %       3: Semi-random start, random trajectories
            %       4: Random start points, random trajectories (ICRA17)
            
            obj.trajectory_mode = mode;
            obj.apply_trajectory_mode();
            
        end
        
        function results = process(obj,method,varargin)
            % ImageRegistration.PROCESS(method,[image_pair])
            %   Process the ImagePair using specified method
            %
            % Inputs
            %   method: a string containing a supported image registration
            %           method i.e. one of {'seqreg','cnn','surf'}
            %   
            %   image_pair (optional): an ImagePair object in place of  
            %                           using set_images() method.
            %
            % Outputs
            %   tform:  the image registration transform returned from
            %           estimateGeometricTransform

            % set current method
            obj.method = method;  
            
            % If an ImagePair obj is passed in, load the appropriate images
            % for the method requested
            if nargin > 2
                
                if isa(varargin{1},'ImagePair')
                    
                    switch obj.method
                        case 'seqreg'
                            set_images(obj,varargin{1}.im1_fullres, ...
                                varargin{1}.im2_fullres, ...
                                varargin{1}.im1_unpad_locnorm, varargin{1}.im2_unpad_locnorm);                            
                            
                        case 'cnn'
                            set_images(obj, ...
                                varargin{1}.im1_fullres_unpad, ...
                                varargin{1}.im2_fullres_unpad, ...
                                varargin{1}.im1_orig, ...
                                varargin{1}.im2_orig);                            
                            
                        case 'surf'
                            set_images(obj,varargin{1}.im1_fullres, ...
                                varargin{1}.im2_fullres, ...
                                varargin{1}.im1_fullres_gray, ...
                                varargin{1}.im2_fullres_gray);
                            
                    end
                else
                    warning(['Second argument not an ImagePair object.' ...
                        ' Either use the set_images() method prior to ' ...
                        'or supply an ImagePair object when calling ' ...
                        'process(). Second argument ignored.']);
                    
                end
            end
            
        
            % check if images have been set
            if obj.images_unset
                results = ImageRegistrationResults();
                warning(['Images have not been set. Please either ' ...
                    'use the set_images() method or provide an ' ...
                    'ImagePair object as second argument to the ' ...
                    'process() method']);
                
                return
            end          

            % perform registration using the requested method
            switch obj.method
                case 'seqreg'
                    obj.seqreg();
                case 'surf'
                    obj.surf();
                case 'cnn'
                    obj.cnn();
                otherwise
                    disp(['Please provide an appropriate method ' ...
                        '"seqreg", "surf" or "cnn"'])
            end
            
            results = obj.get_results();
            
            if obj.visuals && results.registration_successful
                obj.show_matches();
            end
            
            % force setting of new images
            obj.images_unset = true;

        end % end process
        
    end
    
    %% PRIVATE METHODS
    methods (Access = protected)
        
        % General Methods

        function parseInput(obj,in_args)
           p = inputParser;
           p.PartialMatching = false;
           p.CaseSensitive = false;
           % paths
           addParameter(p,'cnn_path','/home/james/ros_ws/src/seqreg_tpp/matlab/caffe/',@ischar);
           addParameter(p,'network','grey_def.prototxt',@ischar);
           addParameter(p,'network_col','colour_def.prototxt',@ischar);
           addParameter(p,'weights','grey_weights.caffemodel',@ischar);
           addParameter(p,'weights_col','colour_weights.caffemodel',@ischar);

           % flags
           addParameter(p,'visuals',false,@islogical);

           % SeqReg Parameters
           addParameter(p,'step_size',20,@isnumeric);
           addParameter(p,'seq_len',100,@isnumeric);
           addParameter(p,'search_fract',0.4,@isnumeric);
           addParameter(p,'border',0.1,@isnumeric);
           addParameter(p,'num_steps',5,@isnumeric);
           addParameter(p,'trajectory_mode',0,@isnumeric);
           addParameter(p,'random_scale',0.2,@isnumeric);
           addParameter(p,'ps',20,@isnumeric);

           % CNN parameters
           addParameter(p,'cnn_window',0.02,@isnumeric);
           addParameter(p,'num_cnn_features',1000,@isnumeric);           
           
           parse(p,in_args{:});
           
           fields = fieldnames(p.Results); 
           
           for i = 1:numel(fields)
               obj.(fields{i}) = p.Results.(fields{i});
           end
           
        end

        function results = get_results(obj)
        % align images for current case

            close all;

            maxdistancefract = 0.01;
            mp = obj.movingPoints;
            fp = obj.fixedPoints;

            % get dimensions of full-res images
            im1fullsz = size(obj.im1_fullres);
            im2fullsz = size(obj.im2_fullres);

            % get dimensions of lo-res images
            xres1 = size(obj.im1,2);
            yres1 = size(obj.im1,1);
            xres2 = size(obj.im2,2);
            yres2 = size(obj.im2,1);

            if ~strcmp(obj.method,'surf')
                mp(:, 1) = mp(:, 1) * (im2fullsz(2) / xres2);
                mp(:, 2) = mp(:, 2) * (im2fullsz(1) / yres2);
                fp(:, 1) = fp(:, 1) * (im1fullsz(2) / xres1);
                fp(:, 2) = fp(:, 2) * (im1fullsz(1) / yres1);
            end

            % find the maximum dimension of the two images
            maxdim = max([im1fullsz(1:2) im2fullsz(1:2)]);

            [tform,inlierPtsOut,inlierPtsIn,tform_status] = estimateGeometricTransform(mp, fp,'similarity','MaxDistance', maxdistancefract * maxdim);
            obj.matchedOriginal = inlierPtsOut;
            obj.matchedDistorted = inlierPtsIn;
            
            results = ImageRegistrationResult();

            if tform_status == 0

                % calculate some output values
                Tinv  = tform.invert.T;
                ss = Tinv(2,1);
                sc = Tinv(1,1);
                
                results.scaleRecovered = sqrt(ss*ss + sc*sc);
                results.thetaRecovered = atan2(ss,sc)*180/pi;
                results.percentPtsUsed = size(inlierPtsOut,1)/size(mp,1);
                results.tform = tform;
                results.tform_status = tform_status;
                
                if obj.test_points_used
                    results.min_pts_used = size(inlierPtsIn,1) >= obj.min_pts_required;
                end
                
                results.registration_successful = true;
                
            else
                
                results.tform_status = tform_status;
                
                if tform_status == 1
                    warning('Not enough matched points matched.');
                else
                    warning('Not enough inliers have been found.');
                end
                
            end


        end % end get_results

        function show_matches(obj)
        % generate a matched point pairs image for a case n and specified
        % method, requires reprocessing entire case but doesn't overwrite
        % existing results

            switch obj.method
                case 'cnn'

                    showMatchedFeatures(obj.im1_fullres, ...
                        obj.im2_fullres,obj.matchedOriginal, ...
                        obj.matchedDistorted,'montage');

                case 'surf'

                    showMatchedFeatures(obj.im1_fullres, ...
                        obj.im2_fullres,obj.matchedOriginal.Location, ...
                        obj.matchedDistorted.Location,'montage');

                case 'seqreg'
                    if obj.seqreg_showallmatches
                        showMatchedFeatures(obj.im1_fullres, ...
                            obj.im2_fullres,obj.matchedOriginal, ...
                            obj.matchedDistorted,'montage');
                    else

                        % This commented code removes matched trajectories
                        % with origins in the padded regions, useful for 
                        % less confusing matched pair visualisations
                        [height,width,~] = size(obj.im1_fullres);

                        suitable_h = [1:height] .* ...
                            prod(~all(permute(obj.im1_fullres,[2 1 3]) ...
                            == 0),3);
                        suitable_w = [1:width] .* ...
                            prod(~all(obj.im1_fullres == 0),3);

                        mO = [];

                        for i = 1:size(obj.matchedOriginal,1)

                            mO = [mO all([any(suitable_h == ...
                                round(obj.matchedOriginal(i,2))) ...
                                any(suitable_w == ...
                                round(obj.matchedOriginal(i,1)))])];
                        end

                        [height,width,~] = size(obj.im2_fullres);

                        suitable_h = [1:height] .* ...
                            prod(~all(permute(...
                            obj.im2_fullres,[2 1 3]) == 0),3);
                        suitable_w = [1:width] .* ...
                            prod(~all(obj.im2_fullres == 0),3);

                        mD = [];

                        for i = 1:size(obj.matchedDistorted,1)

                            mD = [mD all([any(suitable_h == ...
                                round(obj.matchedDistorted(i,2))) ...
                                any(suitable_w == ...
                                round(obj.matchedDistorted(i,1)))])];

                        end

                        ind = mO & mD;

                        mO = obj.matchedOriginal(ind,:);
                        mD = obj.matchedDistorted(ind,:);

                        showMatchedFeatures(obj.im1_fullres, ...
                            obj.im2_fullres,mO,mD,'montage');
                    end
                    
                otherwise
                    fprintf('Unsuitable method supplied\n');

            end

            h = findobj(gcf,'Type','line');
            h(1).LineWidth = 2.0;
            refreshdata
            pause
            close all

        end % end show_matches
        
        % SeqReg Related Methods
        
        function apply_trajectory_mode(obj)
            
            switch obj.trajectory_mode
                
                case 0 % ORIGINAL (ACRA 2016)
                    fprintf('Original trajectories selected\n');
                    obj.random_startpoints = false;
                    obj.semi_random_startpoints = false;
                    obj.random_trajectories = false;
                    obj.fullspread = false;
                    
                case 1 % REGULAR START, RANDOM TRAJECTORY
                    fprintf('Regular start, random trajectories selected\n');
                    obj.random_startpoints = false;
                    obj.semi_random_startpoints = false;
                    obj.random_trajectories = true;
                    obj.fullspread = false;

                case 2 % FULLSPREAD START, RANDOM TRAJECTORY (ICRA 2017)
                    fprintf('Fullspread start, random trajectories selected\n');
                    obj.random_startpoints = false;
                    obj.semi_random_startpoints = false;
                    obj.random_trajectories = true;
                    obj.fullspread = true;

                case 3 % SEMIRANDOM START, RANDOM TRAJECTORY
                    fprintf('Semi-random start, random trajectories selected\n');
                    obj.random_startpoints = false;
                    obj.semi_random_startpoints = true;
                    obj.random_trajectories = true;
                    obj.fullspread = false;

                case 4 % RANDOM START, RANDOM TRAJECTORY (ICRA 2017)
                    fprintf('Random start, random trajectories selected\n');
                    obj.random_startpoints = true;
                    obj.semi_random_startpoints = false;
                    obj.random_trajectories = true;
                    obj.fullspread = true;
                    
            end
            
        end % end apply_trajectory_mode
        
        function seqreg(obj,varargin)
        % perform seqreg image registration

            obj.apply_trajectory_mode();

            obj.fixedPoints = [];
            obj.movingPoints = [];
            obj.traj_strengths = [];
            obj.strongest_match = 0;
            close all
            
            s1 = size(obj.im1);
            s2 = size(obj.im2);

            % add black padding to match other image size
            im1_padding = [max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))];
            im2_padding = [max(0,round((s1(1) - s2(1))/2)), max(0,round((s1(2) - s2(2))/2))];

            % get im dimensions
            imsz = size(obj.im2);
            imsz_query = size(obj.im1);

            % calculate range of image used
            corr_rangex = round(obj.search_fract * imsz(1));
            corr_rangey = round(obj.search_fract * imsz(2));

            if obj.ps > min(imsz) /10
                obj.curr_ps = floor(min(imsz) /10);
            else
                obj.curr_ps = obj.ps;
            end

            if sqrt(2*(obj.seq_len^2)) > (min(imsz) - 2*obj.curr_ps)
                obj.curr_seq_len = sqrt(((min(imsz) - 2*obj.curr_ps)^2)/2);
                obj.curr_step_size = floor(min(imsz) /10);
            else
                obj.curr_seq_len = obj.seq_len;
                obj.curr_step_size = floor(min(imsz) /10);
            end

            if obj.visuals
                close all
                fig1h = figure;
                subplot(1, 2, 1);
                imshow(obj.im1);
                caxis([-2 2]);
                axis on;
                subplot(1, 2, 2);
                imshow(obj.im2);
                caxis([-2 2]);
                axis on;
            end

            % Search start points
            if ~obj.semi_random_startpoints
                if obj.fullspread
                    xrange = (round(obj.border * imsz(2)):obj.curr_step_size:round( (1-obj.border) * imsz(2)));
                    yrange = (round(obj.border * imsz(1)):obj.curr_step_size:round( (1-obj.border) * imsz(1)));
                else
                    xrange = (round(obj.border * imsz(2)):obj.curr_step_size:round( (1-obj.border) * imsz(2) - obj.curr_seq_len));
                    yrange = (round(obj.border * imsz(1)):obj.curr_step_size:round( (1-obj.border) * imsz(1) - obj.curr_seq_len));
                end
            else
                % Semi-random Start Points
                xrange = randi([round(obj.border * imsz(2)),round((1-obj.border) * imsz(2))],1,round((imsz(2) - obj.curr_seq_len) / obj.curr_step_size));
                yrange = randi([round(obj.border * imsz(1)),round((1-obj.border) * imsz(1))],1,round((imsz(1) - obj.curr_seq_len) / obj.curr_step_size));
            end

            % loop through xrange, yrange
            for startx = xrange

                for starty = yrange

                    % could probably be handled differently, consider other
                    % options
                    if obj.random_startpoints
                        startx = randi([round(obj.border * imsz(2)),round((1-obj.border) * imsz(2))],1,1);
                        starty = randi([round(obj.border * imsz(1)),round((1-obj.border) * imsz(1))],1,1);
                    end

                    xp1 = round(startx);
                    yp1 = round(starty);

                    % Random Trajectory and Scale
                    if obj.random_trajectories
                        xp2 = -1;
                        yp2 = -1;
                        while (xp2 < round(obj.border * imsz(2)) || xp2 > round((1-obj.border) * imsz(2))) ||  (yp2 < round(obj.border * imsz(1)) || yp2 > round((1-obj.border) * imsz(1)))
                            scale = (2*obj.random_scale)*rand(1) + (1-obj.random_scale);
                            angle = 2*pi*rand(1);
                            xp2 = startx + round((sqrt(2*obj.curr_seq_len^2)*scale)*sin(angle));
                            yp2 = starty + round((sqrt(2*obj.curr_seq_len^2)*scale)*cos(angle));
                            xp2_unscaled = startx + round((sqrt(2*obj.curr_seq_len^2))*sin(angle));
                            yp2_unscaled = starty + round((sqrt(2*obj.curr_seq_len^2))*cos(angle));
                        end
                    else
                        % Diagonal sequences
                        xp2 = round(startx + obj.curr_seq_len);
                        yp2 = round(starty + obj.curr_seq_len);
                    end

                    % calculate vertical and horizontal stepsize
                    xstepsize = (xp2 - xp1) / (obj.num_steps - 1);
                    ystepsize = (yp2 - yp1) / (obj.num_steps - 1);

                    % init xp and yp
                    xp = xp1;
                    yp = yp1;

                    if obj.random_trajectories_experimental
                        xstepsize_unscaled = (xp2_unscaled - xp1) / (obj.num_steps - 1);
                        ystepsize_unscaled = (yp2_unscaled - yp1) / (obj.num_steps - 1);
                        xp_unscaled = xp1;
                        yp_unscaled = yp1;
                    end

                    % save original step sizes
                    orig_xstepsize = xstepsize;
                    orig_ystepsize = ystepsize;
                    orig_d = norm([xp2 - xp1 yp2 - yp1]);
                    center_angle = atan2(orig_ystepsize, orig_xstepsize);
                    angle_spread = deg2rad(0);
                    angle_step = deg2rad(5);

                    % No range currently applied
                    dmultrange = 1;
%                     dmultrange = 0.98:0.01:1.02;

                    angle_range = center_angle - angle_spread:angle_step:center_angle + angle_spread;

                    % create trajectory
                    ppos = zeros(obj.num_steps*length(angle_range)*length(dmultrange),2);
                    if obj.random_trajectories_experimental
                        ppos_unscaled = zeros(obj.num_steps*length(angle_range)*length(dmultrange),2);
                    end
                    count = 1;

                    % check multiple trajectory lengths (read scales)
                    for dmult = dmultrange

%                         % apply distance multiplier
%                         d = orig_d * dmult;

                        for a = angle_range

%                             xstepsize = d / (obj.num_steps - 1) * cos(a);
%                             ystepsize = d / (obj.num_steps - 1) * sin(a);

                            for z = 1:obj.num_steps
                                ppos(count, :) = [round(xp) round(yp)];
                                if obj.random_trajectories_experimental
                                    ppos_unscaled(count, :) = [round(xp_unscaled) round(yp_unscaled)];
                                    xp_unscaled = xp1 + z * xstepsize_unscaled;
                                    yp_unscaled = yp1 + z * ystepsize_unscaled;
                                end
                                count = count + 1;
                                xp = xp1 + z * xstepsize;
                                yp = yp1 + z * ystepsize;

                            end

                        end
                    end

                    ppos = ppos(~(any(ppos' <= obj.curr_ps)' | (ppos(:,1)' > (imsz(2) - obj.curr_ps))' | (ppos(:,2)' > (imsz(1) - obj.curr_ps))'),:);

                    if isempty(ppos)
                        fprintf('ppos empty!');
                        continue
                    end

                    % Create the difference matrices for each step along the trajectory
                    xchist = zeros(imsz_query(1),imsz_query(2),obj.num_steps);

                    for z = 1:size(ppos,1)

                        % extract region around point on trajectory

                        p2 = obj.im2(ppos(z, 2)-obj.curr_ps:ppos(z, 2)+obj.curr_ps, ppos(z, 1)-obj.curr_ps:ppos(z, 1)+obj.curr_ps);

                        % ensure std != 0?
                        if all(p2 == p2(1))
                            p2(1, 1) = p2(1, 1) + 1;
                        end

                        % Old way doing comparison against whole image
                        if obj.search_fract == 1
                            % compute correlations between region from im2 and entire im1x
                            xc1 = normxcorr2(p2, obj.im1);

                            % crop off edges of correlation image to original size
                            xc1 = xc1(obj.curr_ps + 1: imsz_query(1) + obj.curr_ps, obj.curr_ps + 1: imsz_query(2) + obj.curr_ps);

                            % save correlation image
                            xchist(:, :, z) = xc1;
                        else
                            % New way doing in local region
                            xcsumrng = (max(1, yp1 - corr_rangex):min(imsz(1), yp1 + corr_rangex) );
                            ycsumrng = (max(1, xp1 - corr_rangey):min(imsz(2), xp1 + corr_rangey) );
                            xc_temp = normxcorr2(p2, obj.im1(xcsumrng, ycsumrng));
                            xc2 = zeros(imsz_query);
                            xc_temp = xc_temp(obj.curr_ps+1:size(xc_temp,1)-obj.curr_ps,obj.curr_ps+1:size(xc_temp,2)-obj.curr_ps);
                            xc2([0:size(xc_temp,1) - 1] + xcsumrng(1), [0:size(xc_temp,2) - 1] + ycsumrng(1)) = xc_temp;
                            xchist(:, :, z) = xc2;
                        end

                    end

                    if obj.random_trajectories_experimental
                        ppos = ppos_unscaled;

                    end
%
                    % Stack the images
                    xcsum = zeros(imsz_query);

                    for z = 1:size(ppos,1)

                        xoffset = -(ppos(z, 1) - xp1);
                        yoffset = -(ppos(z, 2) - yp1);

                        x1start = max(1, xoffset + 1);
                        x1finish = min(imsz_query(2) + xoffset, imsz_query(2));
                        y1start = max(1, yoffset + 1);
                        y1finish = min(imsz_query(1) + yoffset, imsz_query(1));

                        x2start = max(1, -xoffset + 1);
                        x2finish = min(imsz_query(2) - xoffset, imsz_query(2));
                        y2start = max(1, -yoffset + 1);
                        y2finish = min(imsz_query(1) - yoffset, imsz_query(1));

                        xcsum(y1start:y1finish, x1start:x1finish) = xcsum(y1start:y1finish, x1start:x1finish) + (xchist(y2start:y2finish, x2start:x2finish, z).^3);

                    end

                    xcsum = xcsum / obj.num_steps;

                    abs_xcsum = abs(xcsum);

                    if obj.visuals
                        drawnow;
                    end

                    [y, i] = max(abs_xcsum(:));

                    [i,j] = ind2sub(size(abs_xcsum), i);

                    best_peak = [i j];

                    % collect matched points
                    obj.fixedPoints = [obj.fixedPoints; xp1 yp1];
                    obj.movingPoints = [obj.movingPoints; best_peak(2) best_peak(1)];

                    obj.traj_strengths = [obj.traj_strengths; y];

                    strongest_local_match = y;
                    obj.strongest_match = max(obj.strongest_match, strongest_local_match);

                    if obj.visuals

                        % Plot figure
                        figure(fig1h);

                        subplot(1,2,2);

                        hold on;

                        plot(obj.fixedPoints(end, 1), obj.fixedPoints(end, 2), 'rx', 'MarkerSize', 2, 'LineWidth', 1);
                        plot([obj.fixedPoints(end, 1) obj.fixedPoints(end, 1) + ppos(end,1) - xp1], [obj.fixedPoints(end, 2) obj.fixedPoints(end, 2) + ppos(end,2) - yp1], 'b-', 'MarkerSize', 5, 'LineWidth', 1);

                        subplot(1,2,1);

                        hold on;

                        plot(obj.movingPoints(end, 1), obj.movingPoints(end, 2), 'bo', 'MarkerSize', max(1, round(strongest_local_match * 200)), 'LineWidth', 1);


                    end

                end

            end

%             obj.movingPoints(:,1) = obj.movingPoints(:,1) + im1_padding(2);
%             obj.movingPoints(:,2) = obj.movingPoints(:,2) + im1_padding(1);

%             if all(size(obj.im1_unpad_locnorm) < size(obj.im2_unpad_locnorm))
                obj.fixedPoints(:,1) = obj.fixedPoints(:,1) + im2_padding(2);
                obj.fixedPoints(:,2) = obj.fixedPoints(:,2) + im2_padding(1);
                obj.im2 = padarray(obj.im2,im2_padding);
%             end

%             obj.im1 = padarray(obj.im1,im1_padding);

            if obj.filter_points
                ind = obj.traj_strengths > median(obj.traj_strengths);
                obj.movingPoints = obj.movingPoints(ind,:);
                obj.fixedPoints = obj.fixedPoints(ind,:);
            end

        end % end seqreg

        % CNN Related Methods

        % set the path to the network definition file
        function set_network(obj,network)

            if ~strcmp('.prototxt',network(end-8:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.network = network;
            obj.load_model();

        end % end set_network
        
        % set the path to the network definition file
        function set_network_col(obj,network)

            if ~strcmp('.prototxt',network(end-8:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.network_col = network;
            obj.load_model();

        end % end set_network
        
        % set the path to the weights file
        function set_weights(obj,weights)

            if ~strcmp('.caffemodel',weights(end-10:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.weights = weights;
            obj.load_model();

        end % end set_weights
        
        % set the path to the weights file
        function set_weights_col(obj,weights)

            if ~strcmp('.caffemodel',weights(end-10:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.weights_col = weights;
            obj.load_model();

        end % end set_weights
        
        % set the caffe model as a preloaded model
        function set_model(obj,model)

            obj.model = model;

        end % end set_model
        
        function load_model(obj)
            
            obj.model_loaded = false;
            
            if ~isempty(obj.weights) && ~isempty(obj.network)

                disp('Loading caffe model...')
                
                try
                    obj.model = caffe.Net([obj.cnn_path '/' obj.network],[obj.cnn_path '/' obj.weights],'test');
                catch
                    disp('Model not loaded.')
                end

            end

            if ~isempty(obj.weights_col) && ~isempty(obj.network_col)

                disp('Loading caffe model...')

                try
                    obj.model_col = caffe.Net([obj.cnn_path '/' obj.network_col],[obj.cnn_path '/' obj.weights_col],'test');
                catch
                    disp('Model not loaded.')
                end

            end
            
            obj.model_loaded = true;

        end % end load_model

        function cnn(obj,varargin)
        % perform image registration with cnn method
        
            if ~obj.model_loaded
                obj.load_model();
            end

            close all

            im1_data = single(obj.im1);
            im2_data = single(obj.im2);

            if length(size(im1_data)) == 2
                im1_data = squeeze(im1_data(:,:,1));
                im1_data = permute(im1_data, [2, 1]); % permute width and height
                shape1 = size(im1_data);
                shape1 = [shape1 1 1];
                im1_data = (im1_data - mean2(im1_data)) / std2(im1_data);
                if any(obj.model.blobs('data').shape ~= shape1)
                    obj.model.blobs('data').reshape(shape1);
                end

                obj.model.blobs('data').set_data(im1_data);

                obj.model.forward_prefilled();

                im1_conv1 = obj.model.blobs('conv1').get_data();
                
                im1_pad = obj.model.layers('conv1_g').params(1).shape;
                im1_pad = floor(im1_pad(1) / 2);

            else
                std_ = reshape([std2(im1_data(:,:,1)) std2(im1_data(:,:,2)) std2(im1_data(:,:,3))],[1 1 3]);
                mean_ = reshape([mean2(im1_data(:,:,1)) mean2(im1_data(:,:,2)) mean2(im1_data(:,:,3))],[1 1 3]);
                im1_data = im1_data(:, :, [3, 2, 1]); % convert from RGB to BGR
                im1_data = permute(im1_data, [2, 1, 3]); % permute width and height
                shape1 = size(im1_data);
                shape1 = [shape1 1];

                im1_data = bsxfun(@minus, im1_data, mean_);
                im1_data = bsxfun(@rdivide, im1_data, std_);

                if any(obj.model_col.blobs('data').shape ~= shape1)
                    obj.model_col.blobs('data').reshape(shape1);
                end

                obj.model_col.blobs('data').set_data(im1_data);

                obj.model_col.forward_prefilled();

                im1_conv1 = obj.model_col.blobs('conv1').get_data();
                
                im1_pad = obj.model_col.layers('conv1').params(1).shape;
                im1_pad = floor(im1_pad(1) / 2);

            end

            if length(size(im2_data)) == 2
                im2_data = squeeze(im2_data(:,:,1));
                im2_data = permute(im2_data, [2, 1]); % permute width and height
                shape2 = size(im2_data);
                shape2 = [shape2 1 1];

                if any(obj.model.blobs('data').shape ~= shape2)
                    obj.model.blobs('data').reshape(shape2);
                end
                im2_data = (im2_data - mean2(im2_data)) / std2(im2_data);

                obj.model.blobs('data').set_data(im2_data);

                obj.model.forward_prefilled();

                im2_conv1 = obj.model.blobs('conv1').get_data();
                
                im2_pad = obj.model.layers('conv1_g').params(1).shape;
                im2_pad = floor(im2_pad(1) / 2);

            else

                std_ = reshape([std2(im2_data(:,:,1)) std2(im2_data(:,:,2)) std2(im2_data(:,:,3))],[1 1 3]);
                mean_ = reshape([mean2(im2_data(:,:,1)) mean2(im2_data(:,:,2)) mean2(im2_data(:,:,3))],[1 1 3]);
                im2_data = im2_data(:, :, [3, 2, 1]); % convert from RGB to BGR
                im2_data = permute(im2_data, [2, 1, 3]); % permute width and height
                shape2 = size(im2_data);
                shape2 = [shape2 1];
                if any(obj.model_col.blobs('data').shape ~= shape2)
                    obj.model_col.blobs('data').reshape(shape2);
                end

                im2_data = bsxfun(@minus, im2_data, mean_);
                im2_data = bsxfun(@rdivide, im2_data, std_);

                obj.model_col.blobs('data').set_data(im2_data);

                obj.model_col.forward_prefilled();

                im2_conv1 = obj.model_col.blobs('conv1').get_data();
                
                im2_pad = obj.model_col.layers('conv1').params(1).shape;
                im2_pad = floor(im2_pad(1) / 2);

            end

            im1_conv1 = sum(im1_conv1,3);
            im2_conv1 = sum(im2_conv1,3);

            im1_conv1 = (im1_conv1 - min(im1_conv1(:))) / max(im1_conv1(:));
            im2_conv1 = (im2_conv1 - min(im2_conv1(:))) / max(im2_conv1(:));

            im1_conv1 = im1_conv1';
            im2_conv1 = im2_conv1';

            n1 = round(size(im1_conv1) * obj.cnn_window);
            n2 = round(size(im2_conv1) * obj.cnn_window);
            rad = max(max(n1,n2));

            s1 = size(im1_conv1);
            s2 = size(im2_conv1);

            points1 = zeros([obj.num_cnn_features,2]);
            points2 = zeros([obj.num_cnn_features,2]);

            im1_conv1_ = im1_conv1;
            im2_conv1_ = im2_conv1;

            for i = 1:obj.num_cnn_features
                [~,m] = max(im1_conv1(:));
                [a,b] = ind2sub(size(im1_conv1),m);
                points1(i,:) = [a b];
                im1_conv1(max(1,a-n1(1)):min(s1(1),a+n1(1)),max(1,b-n1(2)):min(s1(2),b+n1(2))) = 0;
                [~,m] = max(im2_conv1(:));
                [a,b] = ind2sub(size(im2_conv1),m);
                points2(i,:) = [a b];
                im2_conv1(max(1,a-n2(1)):min(s2(1),a+n2(1)),max(1,b-n2(2)):min(s2(2),b+n2(2))) = 0;
            end

            if obj.visuals
                close all
                imshowpair(im1_conv1_,im2_conv1_,'method','montage');
            end

            [featuresOriginal, validPtsOriginal] = extractFeatures(im1_conv1_,  points1, 'method', 'Block','BlockSize',rad*2 + 1);
            [featuresDistorted, validPtsDistorted] = extractFeatures(im2_conv1_, points2, 'method', 'Block','BlockSize',rad*2 + 1);

            % Match features by using their descriptors.
            indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

            % Retrieve locations of corresponding points for each image.

            % padding due to reduced size of feature map compared to input
            % image
            obj.movingPoints = validPtsOriginal(indexPairs(:,1),:) + im1_pad; 
            obj.fixedPoints = validPtsDistorted(indexPairs(:,2),:) + im2_pad;

        end % end cnn
        
        % SURF Related Methods

        function surf(obj,varargin)
        % perform image registration with SURF features

            % Detect features in both images.

            ptsOriginal  = detectSURFFeatures(obj.im1);
            ptsDistorted = detectSURFFeatures(obj.im2);

            % Extract feature descriptors.
            [featuresOriginal, validPtsOriginal] = extractFeatures(obj.im1,  ptsOriginal);
            [featuresDistorted, validPtsDistorted] = extractFeatures(obj.im2, ptsDistorted);

            % Match features by using their descriptors.
            indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

            % Retrieve locations of corresponding points for each image.
            obj.movingPoints = validPtsOriginal(indexPairs(:,1));
            obj.fixedPoints = validPtsDistorted(indexPairs(:,2));

        end % end surf

    end

end
