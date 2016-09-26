classdef NASApipeline < handle
    %NASAPIPELINE Summary of this class goes here
    %   Detailed explanation goes here
%%
    properties
        known_bad_cases
        test_cases
        curr_case
        weights
        weights_col
        tform
        tform_status
        cnn_window = 0.02
        network = '/home/james/ros_ws/src/seqslam_tpp/matlab/caffe/deploy_conv1.prototxt'
        network_col = '/home/james/ros_ws/src/seqslam_tpp/matlab/caffe/deploy_conv1_3.prototxt'
        model = '/home/james/ros_ws/src/seqslam_tpp/matlab/caffe/imagenet_grey_weights.caffemodel'
        model_col = '/home/james/ros_ws/src/seqslam_tpp/matlab/caffe/bvlc_reference_caffenet.caffemodel'
        method
        patch_norm = 1
        maxdim = 400
        qsf = 1          % ? scale factor?
        step_size = 20   % step size
        seq_len = 100    % seq length
        minstd = 0.1;   % min std for use in patch normalisation
        ps
        nps
        trans_limit = 0.15
        im1
        im2
        im1_test
        im2_test
        im1_unpad
        im2_unpad
        im1_orig
        im2_orig
        im1_fullres
        im2_fullres
        im1_fullres_unpad
        im2_fullres_unpad
        im1_fullres_gray
        im2_fullres_gray
        im1_reg
        im2_reg
        im1_registered
        im2_registered
        fixedPoints
        movingPoints
        search_fract = 0.1; % region of image to use
        xres
        yres
        border = 0.1;   % ignore border cases
        num_steps = 5; %no. trajectory steps
        visuals = false
        strongest_match = 0; % initialise strogest match, not actually used?
        save_dir = '/home/james/Dropbox/NASA/SeqSLAM/'
        ros_gifdir = '/home/james/Dropbox/NASA/baxter_experiments/gifs/'
        gifdir = '/home/james/datasets/pixl-watson/'
        results = {}
        test_totals
        type_table
        matchedOriginal
        matchedDistorted
        InitialImage
        watsonOrig
        pixlImage
        ros_scale
        curr_trial
        seqslamSrv
        save_images = false
        gpu_locnorm
        gpu_out1
        gpu_out2
        gpu_rgb2gray
        gpu_processing = false
        gpu_processing_fail = false
        gpu_devel_mode = false
        gpu_trajshift
        gpu_xchist
        im1_gpu
        im2_gpu
        gpu_normxcorr2
        GPU
        block_size_num_steps
        block_size1
        save_string
        gpu_sad2
        scales
        random_startpoints = false
        semi_random_startpoints = false
        random_trajectories = false
        fullspread = false
        trajectory_mode = 0
        random_trajectories_experimental = false
        random_scale = 0.2
        traj_strengths = [];
        filter_points = false;
        test_points_used = false;
        fraction_pts_used ;
        im1_padding
        im2_padding
        im1_unpad_double
        im1_unpad_locnorm
        im2_unpad_double
        im2_unpad_locnorm
        min_pts_used
        min_pts_required = 4;
        min_fraction_pts_used = 0.1
        curr_seq_len
        curr_step_size
        gif_visuals = false
        pixl_information
        watson_information
        curr_ps
        ros_initialised
        results_publisher
    end
%%
    methods

%%
        function obj = NASApipeline(varargin)
        % NASApipeline constructor
        % Supply a vector of structures with the fields:
        %   Image1      A structure containing information about Image1
        %   Image2      A structure containing information about Image2
        %   context     The sample number
        %   differences A structure specifying translation, lighting,
        %               scale and multimodal differences
        %   seqslam     An empty structure for storing SeqSLAM results
        %   cnn         An empty structure for storing CNN results
        %   surf        An empty structure for storing SURF results

            if nargin > 0
                obj.test_cases = varargin{1};
                obj.curr_case = 1;
            end

            obj.init();

        end % end NASApipeline
%%
        function init(obj)
        % initialise the object, can be used to reinitialise when reloading
            close all
            rng('shuffle');
            
            obj.gpu_processing = false;
            
            for i=1:gpuDeviceCount
                if parallel.gpu.GPUDevice.isAvailable(i)
                    obj.GPU = parallel.gpu.GPUDevice.getDevice(i);
                    obj.gpu_processing = true;
                    break;
                end
            end

            % initialise ros and services
            obj.init_ros();
            
            if obj.gpu_processing
                ptx_path = '/home/james/co/SeqSLAM_GPU/SeqSLAM_kernel.ptx';
                cu_path = '/home/james/co/SeqSLAM_GPU/SeqSLAM_kernel.cu';

                obj.block_size1 = floor(sqrt(obj.GPU.MaxThreadsPerBlock));
                obj.block_size_num_steps = floor(sqrt(obj.GPU.MaxThreadsPerBlock / obj.num_steps));

                obj.gpu_locnorm = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z10local_normPKfPfiiif');
                obj.gpu_locnorm.ThreadBlockSize = [obj.block_size1 obj.block_size1];            

%                 obj.gpu_rgb2gray = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z18bgr_to_gray_kernelPKhPfiiii');
%                 obj.gpu_rgb2gray.ThreadBlockSize = [obj.block_size1 obj.block_size1];
% 
%                 obj.gpu_trajshift = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z19trajectory_shiftingPKfPfPKiS3_ii');
%                 obj.gpu_trajshift.ThreadBlockSize = [obj.block_size_num_steps obj.block_size_num_steps,obj.num_steps];
% 
%                 obj.gpu_normxcorr2 = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z10normXcorr2PKfPfS0_S0_S0_iii');
%                 obj.gpu_normxcorr2.ThreadBlockSize = [obj.block_size_num_steps obj.block_size_num_steps,obj.num_steps];
% 
%                 obj.gpu_sad2 = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z4sad2PKfPfS0_Piiii');
%                 obj.gpu_sad2.ThreadBlockSize = [obj.block_size_num_steps obj.block_size_num_steps,obj.num_steps];

                wait(obj.GPU);
                
                % set caffe to use gpu also
                caffe.set_mode_gpu();
            end
            
            % initialise variables
            obj.ps = 20 * obj.qsf;  % Radius of patch to be correlated
            obj.nps = 5 * obj.qsf;	% Patch normalization range

            % intialise caffe            
            obj.load_model();
            
            obj.results_publisher = ResultsPublisher(obj);

        end % end init
%%
        function init_ros(obj)
            
            if obj.ros_initialised
                rosshutdown;                
            end
            
            rosinit;
            obj.ros_initialised = true;            
            obj.seqslamSrv = rossvcserver('/seqslam_tpp/seqslam','seqslam_tpp/MATLABSrv',@obj.seqslamCallback);
            
        end
        
%%
        % accepts custom srv type
        function res = seqslamCallback(obj,~,req,res)
            
            fprintf('%s started\n',req.Method.Data);

            [obj.InitialImage,~] = readImage(req.InitialImage);
            
            obj.save_string = num2str(req.SecondaryImage.Header.Stamp.Sec);

            obj.watsonOrig = obj.InitialImage;
            
            obj.ros_scale = req.Scales.Data';

            obj.curr_trial = obj.curr_trial + 1;
            
            fprintf('Initial registration with SeqSLAM to estimate new pose\n');
            
            [obj.pixlImage,~] = readImage(req.SecondaryImage);
            
            if obj.visuals                
                close all
                imshowpair(obj.InitialImage,obj.pixlImage,'method','montage');
            end
            
            % perform image registration
            obj.process(req.Method.Data,'save',false);

            obj.ros_save_im();
            
            Tinv = obj.tform.invert.T;
            
            ss = Tinv(2,1);
            sc = Tinv(1,1);
            
            scaleRecovered = sqrt(ss*ss + sc*sc);
            thetaRecovered = atan2(ss,sc)*180/pi;
            
            fprintf('Initial Registration\n\tEstimated Scale:\t%0.3f\n\tEstimated Rotation:\t%0.3f\n\tTrans X: %0.3f\n\tTrans Y: %0.3f\n\tPoints Used: %s\n\tFraction Points: %0.3f\n',scaleRecovered,thetaRecovered,obj.tform.T(3,2),obj.tform.T(3,1),bool2str(obj.min_pts_used),obj.fraction_pts_used);
            
            if abs(1 - scaleRecovered) < 2.0 && abs(thetaRecovered) < 10 && obj.fraction_pts_used > obj.min_fraction_pts_used && obj.min_pts_used
                res.Results.Data = [scaleRecovered thetaRecovered obj.tform.T(3,2) obj.tform.T(3,1)];
                res.Success.Data = true;
                res.Message.Data = 'Registration with SeqSLAM succeeded!';
                fprintf('Registration with SeqSLAM succeeded!\n');
            else
                res.Results.Data = [0.0 0.0 0.0 0.0];
                res.Success.Data = false;
                res.Message.Data = 'Initial registration with SeqSLAM failed!';
                fprintf('Registration with SeqSLAM failed!\n');
            end
            
            if obj.visuals
                close all
                imshowpair(obj.im1_registered,obj.im2_registered)
            end
            
            obj.save_prog();

        end
%%
        function toggle_vis(obj)
        % toggle visuals flag which can be used throughout for
        % visualising alignments

            obj.visuals = ~obj.visuals;

            if obj.visuals
                fprintf('Visuals on.\n');
            else
                fprintf('Visuals off.\n');
            end

        end % end toggle_vis
%%
        function toggle_save(obj)
        % toggle visuals flag which can be used throughout for
        % visualising alignments

            obj.save_images = ~obj.save_images;

            if obj.visuals
                fprintf('Save images on.\n');
            else
                fprintf('Save_images off.\n');
            end

        end % end toggle_save
        
        function load_csv_data(obj,filenames)
            ftoread = '/home/james/datasets/pixl-watson/pixl.csv';
            fid = fopen(ftoread);
            fgetl(fid);
            obj.pixl_information = textscan(fid, '%f%s%f%f%s', 'Delimiter',','); % you will need to change the number   of values to match your file %f for numbers and %s for strings.
            fclose (fid);
            ftoread = '/home/james/datasets/pixl-watson/watson.csv';
            fid = fopen(ftoread);
            fgetl(fid);
            obj.watson_information = textscan(fid, '%f%s%s%s%f', 'Delimiter',','); % you will need to change the number   of values to match your file %f for numbers and %s for strings.
            fclose (fid);        
            
        end
%%
        function process(obj,method,varargin)
        % process the current set case using specified method

            % set current method
            obj.method = method;
            
            % parse input parameters
            for i = 1 : 2 : length(varargin)
                switch varargin{i}
                    case 'save'
                        obj.save_images = varargin{i+1};
                    otherwise
                        obj.save_images = false;
                end
            end

            % determine if path normalisation required
            if ~isempty(strfind(obj.method,'seqslam'))
                obj.patch_norm = 1;
            else
                obj.patch_norm = 0;
            end

            % load and preprocess images
            %tic
            if isempty(strfind(obj.method,'ros_'))
                obj.im_load();
            else
                obj.ros_imload();
            end
%             fprintf('Image Preprocessing Time: %0.4f seconds\n',toc)

            % perform registration through the appropriate method
            %t_total_start = tic;
            switch obj.method
                case {'seqslam', 'ros_seqslam'}
                    obj.seqslam()
                case {'seqslam_high_res'}
                    obj.seqslam()
                case 'new_seqslam'
                    obj.new_seqslam()
                case {'surf','ros_surf'}
                    obj.surf()
                case 'cnn'
                    obj.cnn()
                case 'ros_cnn'
                    obj.ros_cnn()
                otherwise
                    disp('Please provide an appropriate method "seqslam", "surf" or "cnn"')
            end
            
%             t_total = toc - t_total_start;
%             fprintf('Method Time: %0.4f seconds\n',t_total)

            % estimate transform
            if isempty(strfind(obj.method,'ros_'))
                obj.align_im();
            else
                obj.ros_align_im();
            end

            % generate and save gif
            if obj.save_images && obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).match
                obj.save_im();
            elseif obj.save_images && ~isempty(strfind(obj.method,'ros_'))
                obj.ros_save_im();
            end

        end % end process

%%
        function ros_save_im(obj)
        % generate gif using the estimated transform

            im1f = obj.im1_fullres;
            im2f = obj.im2_fullres;

            % create number of frames, desired dimension
            nframes = 10;
            desdim = 800;

            % different methods use different fixed and moving images,
            % should update this to make consistent
            [xLimitsOut,yLimitsOut] = outputLimits(obj.tform, [1 size(im1f,2)], [1 size(im1f,1)]);

            % Find the minimum and maximum output limits
            xMin = min([1; xLimitsOut(:)]);
            xMax = max([size(im2f,2); xLimitsOut(:)]);
            yMin = min([1; yLimitsOut(:)]);
            yMax = max([size(im2f,1); yLimitsOut(:)]);

            % Width, height and limits of outView
            width  = round(xMax - xMin);
            height = round(yMax - yMin);
            xLimits = [xMin xMax];
            yLimits = [yMin yMax];
            outView = imref2d([height width], xLimits, yLimits);

            % apply identity transform to im2 to match size
            obj.im2_registered = imwarp(im2f, affine2d(eye(3)), 'OutputView', outView);

            % apply estimated transform to im1
            registered = imwarp(im1f, obj.tform, 'OutputView', outView);

            % remove any excess padding from both images
            mask = obj.im2_registered ==0 & registered == 0;
            mask = prod(mask,3);
            mask_test = double(repmat(all(mask'),[size(mask,2),1]))' | double(repmat(all(mask),[size(mask,1),1]));
            obj.im2_registered = obj.im2_registered(~all(mask_test'),~all(mask_test),:);
            registered = registered(~all(mask_test'),~all(mask_test),:);

            % resize to max dimension of desdim
            max_dim = max(size(registered));
            r_dim = [NaN NaN];
            r_dim(size(registered) == max_dim) = desdim;
            obj.im1_registered = imresize(registered,r_dim);
            obj.im2_registered = imresize(obj.im2_registered,r_dim);

%             % Generate gif
%             outstring = sprintf('%s_%s',obj.method,obj.save_string);
% 
%             if ~isdir(obj.ros_gifdir)
%                 mkdir(obj.ros_gifdir);
%             end
% 
%             framename = sprintf('%s%s.gif',obj.ros_gifdir,outstring);
% 
%             start_end_delay = 0.5;
%             normal_delay = 2.0 / nframes;
% 
%             if length(size(obj.im1_orig)) == 2
%                 third_dim = 1;
%             else
%                 third_dim = 3;
%             end
% 
%             for i = 1:nframes
% 
%                 if i == 1 || i == (nframes) / 2 + 1
%                     fdelay = start_end_delay;
%                 else
%                     fdelay = normal_delay;
%                 end
% 
%                 im1_fract = abs( (0.5 * nframes - (i - 1)) / (0.5 * nframes - 1));
% 
%                 % Directional fade
%                 if third_dim == 1
%                     [imind1,cm1] = rgb2ind( repmat(uint8(im1_fract * obj.im2_registered + (1 - im1_fract) * obj.im1_registered), [1 1 3]), 256);
%                 else
%                     [imind1,cm1] = rgb2ind( uint8(im1_fract * obj.im2_registered + (1 - im1_fract) * obj.im1_registered), 256);
%                 end
% 
%                 if i == 1
%                     imwrite(imind1, cm1, framename, 'gif', 'Loopcount', inf, 'DelayTime', fdelay);
%                 else
%                     imwrite(imind1, cm1, framename, 'gif', 'WriteMode', 'append', 'DelayTime', fdelay);
%                 end
% 
%             end

        end

%%
        function save_im(obj)
        % generate gif using the estimated transform

            % test if the current case has successfully estimated a
            % transform
            if ~obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).match
                return
            end

            % depending on method, use padded or unpadded image
            if strcmp(obj.method,'cnn')
                im1f = obj.im1_fullres_unpad;
                im2f = obj.im2_fullres_unpad;
            else
                im1f = obj.im1_fullres;
                im2f = obj.im2_fullres;
            end

            % create number of frames, desired dimension
            nframes = 10;
            desdim = 800;

            % different methods use different fixed and moving images,
            % should update this to make consistent
            if ~strcmp(obj.method,'cnn')
                [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(im2f,2)], [1 size(im2f,1)]);

                % Find the minimum and maximum output limits
                xMin = min([1; xLimitsOut(:)]);
                xMax = max([size(im1f,2); xLimitsOut(:)]);
                yMin = min([1; yLimitsOut(:)]);
                yMax = max([size(im1f,1); yLimitsOut(:)]);

            else
                [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(im1f,2)], [1 size(im1f,1)]);

                % Find the minimum and maximum output limits
                xMin = min([1; xLimitsOut(:)]);
                xMax = max([size(im2f,2); xLimitsOut(:)]);
                yMin = min([1; yLimitsOut(:)]);
                yMax = max([size(im2f,1); yLimitsOut(:)]);

            end

            % Width, height and limits of outView
            width  = round(xMax - xMin);
            height = round(yMax - yMin);
            xLimits = [xMin xMax];
            yLimits = [yMin yMax];
            outView = imref2d([height width], xLimits, yLimits);

            % apply identity transform to im2 to match size
            obj.im2_registered = imwarp(im2f, affine2d(eye(3)), 'OutputView', outView);

            % apply estimated transform to im1
            registered = imwarp(im1f, obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, 'OutputView', outView);

            % remove any excess padding from both images
            mask = obj.im2_registered == 0 & registered == 0;
            mask = prod(mask,3);
            mask_test = double(repmat(all(mask'),[size(mask,2),1]))' | double(repmat(all(mask),[size(mask,1),1]));
            obj.im2_registered = obj.im2_registered(~all(mask_test'),~all(mask_test),:);
            registered = registered(~all(mask_test'),~all(mask_test),:);

            % resize to max dimension of desdim
            max_dim = max(size(registered));
            r_dim = [NaN NaN];
            r_dim(size(registered) == max_dim) = desdim;
            obj.im1_registered = imresize(registered,r_dim);
            obj.im2_registered = imresize(obj.im2_registered,r_dim);

            % Generate gif
            outstring = sprintf('%s_%i',obj.method,obj.curr_case);

            if ~isdir([obj.gifdir obj.method num2str(obj.trajectory_mode) '/' '/'])
                mkdir([obj.gifdir obj.method num2str(obj.trajectory_mode) '/' '/']);
            end

            framename = sprintf('%s%s.gif',[obj.gifdir obj.method num2str(obj.trajectory_mode) '/'],outstring)

            start_end_delay = 0.5;
            normal_delay = 2.0 / nframes;

            if length(size(obj.im1_registered)) == 2
                third_dim = 1;
            else
                third_dim = 3;
            end

            for i = 1:nframes

                if i == 1 || i == (nframes) / 2 + 1
                    fdelay = start_end_delay;
                else
                    fdelay = normal_delay;
                end

                im1_fract = abs( (0.5 * nframes - (i - 1)) / (0.5 * nframes - 1));

                % Directional fade
                if third_dim == 1
                    [imind1,cm1] = rgb2ind( repmat(uint8(im1_fract * obj.im2_registered + (1 - im1_fract) * obj.im1_registered), [1 1 3]), 256);
                else
                    [imind1,cm1] = rgb2ind( uint8(im1_fract * obj.im2_registered + (1 - im1_fract) * obj.im1_registered), 256);
                end

                if i == 1
                    imwrite(imind1, cm1, framename, 'gif', 'Loopcount', inf, 'DelayTime', fdelay);
                else
                    imwrite(imind1, cm1, framename, 'gif', 'WriteMode', 'append', 'DelayTime', fdelay);
                end

            end

            % open gif externally of visuals toggled on
            if obj.gif_visuals
                unix(['gnome-open ' framename]);
            end

            % regenerate the results page for this case
            obj.results_publisher = ResultsPublisher(obj);
            obj.results_publisher.generate_html();

        end
%%
        function show_matches(obj,n,method)
        % generate a matched point pairs image for a case n and specified
        % method, requires reprocessing entire case but doesn't overwrite
        % existing results

            if ~strcmp(obj.method,'ros_seqslam') && ~strcmp(obj.method,'ros_surf')
                obj.patch_norm = 0;

                obj.curr_case = n;

                obj.method = method;

                if strcmp(obj.method,'seqslam')
                    obj.patch_norm = 1;
                end

                obj.im_load();
            end

            switch obj.method
                case 'cnn'
                    obj.cnn();

                    obj.align_im2();

                    showMatchedFeatures(obj.im1_fullres_unpad, ...
                        obj.im2_fullres_unpad,obj.matchedOriginal, ...
                        obj.matchedDistorted,'montage');

                case 'surf'
                    obj.surf();

                    obj.align_im2();

                    showMatchedFeatures(obj.im1_fullres, ...
                        obj.im2_fullres,obj.matchedOriginal.Location, ...
                        obj.matchedDistorted.Location,'montage');

                % in order for better interpretation, only the matched
                % points residing in both image are shown
                case 'seqslam'
                    obj.set_trajectory_mode();
                    obj.seqslam();
                    pause
                    obj.align_im2();

                    [height,width,~] = size(obj.im1_fullres);

                    suitable_h = [1:height] .* ...
                        prod(~all(permute(obj.im1_fullres,[2 1 3]) == 0),3);
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

                case 'ros_seqslam'

                    [height,width,~] = size(obj.im1_fullres);

                    suitable_h = [1:height] .* ...
                        prod(~all(permute(obj.im1_fullres,[2 1 3]) == 0),3);
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

                case 'ros_surf'

                    showMatchedFeatures(obj.im1_fullres, ...
                        obj.im2_fullres,obj.matchedOriginal.Location, ...
                        obj.matchedDistorted.Location,'montage');
                otherwise
                    fprintf('Unsuitable method supplied\n');

            end

            h = findobj(gcf,'Type','line');
            h(1).LineWidth = 2.0;
            refreshdata
            pause
            close all

        end % end show_matches
%%
        function save_prog(obj)
        % save the object to backup results

            d = clock;
            nasa = obj;
            save(sprintf('%snasa-%i%02i%02i.mat',obj.save_dir,d(1),d(2),d(3)),'nasa');

        end
        
%%
        function summ_testing(obj,method)
        % summarise test results for a particular method

            obj.method = method;

            counts = zeros(size(obj.type_table,1),1);
            total_counts = zeros(size(obj.type_table,1),1);

            for i = 1:length(obj.test_cases)

                if any(obj.known_bad_cases == i)

                    continue

                end

                code = [obj.test_cases(i).Image1.mcc ...
                    obj.test_cases(i).Image2.mcc ...
                    obj.test_cases(i).differences.translation ...
                    obj.test_cases(i).differences.lighting ...
                    obj.test_cases(i).differences.scale ...
                    obj.test_cases(i).differences.multimodal];

                ind = find(ismember(obj.type_table,code,'rows'));

                total_counts(ind) = total_counts(ind) + 1;
                if strcmp(obj.method,'seqslam')
                    counts(ind) = counts(ind) + obj.test_cases(i).([obj.method num2str(obj.trajectory_mode)]).match;
                else
                    counts(ind) = counts(ind) + obj.test_cases(i).(obj.method).match;
                end
            end

            obj.results.([obj.method num2str(obj.trajectory_mode)]) = counts;

            obj.test_totals = total_counts;

        end % end summ_testing
%%
        function test_allcasesmodes(obj,varargin)
        % test all or some cases using all methods

            if nargin > 1
                a = varargin{1};
                b = varargin{2};
            else
                a = 1;
                b = length(obj.test_cases);
            end

            if b < a
                c = -1;
            else
                c = 1;
            end

            for i = a:c:b

                obj.curr_case = i;

                obj.patch_norm = 0;

                obj.im_load();
                
                tic
                obj.method = 'cnn';
                obj.cnn();
                obj.align_im();
                obj.save_im();
                fprintf('CNN Time: %0.4f\n',toc)
                
                tic
                obj.method = 'surf';
                obj.surf();
                obj.align_im();
                obj.save_im();
                fprintf('SURF Time: %0.4f\n',toc)

                obj.patch_norm = 1;
                obj.im_load();
                
                tic
                obj.method = 'seqslam';
                obj.seqslam();
                obj.align_im();
                obj.save_im();
                fprintf('SeqSLAM Time: %0.4f\n',toc)
                
                obj.save_prog();

            end

        end
        
%%
        function set_trajectory_mode(obj)
            switch obj.trajectory_mode
                case 0 % ORIGINAL
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

                case 2 % FULLSPREAD START, RANDOM TRAJECTORY
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

                case 4 % RANDOM START, RANDOM TRAJECTORY
                    fprintf('Random start, random trajectories selected\n');
                    obj.random_startpoints = true;
                    obj.semi_random_startpoints = false;
                    obj.random_trajectories = true;
                    obj.fullspread = true;
            end
        end
%%
        function seqslam(obj,varargin)
        % perform seqslam image registration
        
            obj.set_trajectory_mode();

            obj.fixedPoints = [];
            obj.movingPoints = [];
            obj.traj_strengths = [];
            obj.strongest_match = 0;
            close all
            
%             obj.im1 = obj.im1(obj.im1_padding(1):end - obj.im1_padding(1),obj.im1_padding(2):end - obj.im1_padding(2));
%             obj.im1 = obj.im1_unpad_locnorm;
%             if all(size(obj.im1_unpad_locnorm) < size(obj.im2_unpad_locnorm))
            obj.im2 = obj.im2_unpad_locnorm;
%             end
            
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
                % display resized images
                fig1h = figure;
                subplot(1, 2, 1);
%                 imshow(obj.im1_orig);
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
            
            % counter for display correspondances
            lastnumberc = 1;
            
            if obj.gpu_processing || obj.gpu_devel_mode
                xchist_gpu = zeros([imsz(1),imsz(2)*obj.num_steps],'single','gpuArray');
                gpu_xchist_cubed = zeros([imsz(1),imsz(2)*obj.num_steps],'single','gpuArray');
                im2_regions = zeros([obj.curr_ps*2+1,(obj.curr_ps*2+1)*obj.num_steps],'single','gpuArray');
                im2_region_mean = zeros(obj.num_steps,'single','gpuArray');
                im2_region_std = zeros(obj.num_steps,'single','gpuArray');
                im1_padded = gpuArray(single(padarray(obj.im1,[obj.curr_ps obj.curr_ps])));
                im_out = zeros([imsz(1)+2*obj.curr_ps, (imsz(2)+2*obj.curr_ps)*obj.num_steps],'single','gpuArray');
                im_out_sized = zeros([imsz(1)+2*obj.curr_ps, imsz(2)+2*obj.curr_ps],'single','gpuArray');
                step_x = zeros(obj.num_steps,'int32','gpuArray');
                step_y = zeros(obj.num_steps,'int32','gpuArray');
            end
            
            % loop through xrange, yrange
            for startx = xrange
                
                for starty = yrange
                    
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

                        % apply distance multiplier
                        d = orig_d * dmult;

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
                    
                    if obj.gpu_processing || obj.gpu_devel_mode
                        xchist_gpu(:) = 0;
                        gpu_xchist_cubed(:) = 0;
                        im2_regions(:) = 0;
                        im2_region_mean(:) = 0;
                        im2_region_std(:) = 0;
                        im_out(:) = 0;
                        im_out_sized(:) = 0;
                    end
                    
                    if obj.gpu_devel_mode
                        
                        im2_regions = zeros(obj.curr_ps*2+1,(obj.curr_ps*2+1)*size(ppos,1));
                        
                        im2_region_mean = zeros(obj.num_steps);
                        im2_region_std = zeros(obj.num_steps);
                        
                        for z = 1:size(ppos,1)

                            % extract region around point on trajectory
                            p2 = obj.im2(ppos(z, 2)-obj.curr_ps:ppos(z, 2)+obj.curr_ps, ppos(z, 1)-obj.curr_ps:ppos(z, 1)+obj.curr_ps);
                            
                            if all(p2 == p2(1))
                                p2(1, 1) = p2(1, 1) + 1;
                            end
                            
                            im2_region_mean(z) = single(mean(p2(:)));
                            im2_region_std(z) = single(std(p2(:)));
                            
                            im2_regions(:,(z-1)*(obj.curr_ps*2+1)+1:z*(obj.curr_ps*2+1)) = single(p2);
                            
                        end
                        
                        height = imsz(1) + 2*obj.curr_ps;
                        width = imsz(2) + 2*obj.curr_ps;
                        
                        obj.gpu_normxcorr2.GridSize = [ceil(height / obj.block_size_num_steps),ceil(width / obj.block_size_num_steps)];
                        
                        im_out = feval(obj.gpu_normxcorr2,im1_padded,im_out,im2_regions,im2_region_mean,im2_region_std,width,height,obj.curr_ps);
                        
                        for i=1:obj.num_steps
                            xchist_gpu(:,(i-1)*imsz(2)+1:i*imsz(2)) = im_out(obj.curr_ps+1:height-obj.curr_ps,(i-1)*width+obj.curr_ps+1:i*width-obj.curr_ps);
                        end
                    
                    elseif obj.gpu_processing_fail % GPU SAD implementation
                        
                        im2_regions = zeros(obj.curr_ps*2+1,(obj.curr_ps*2+1)*obj.num_steps);
                        
                        for z = 1:obj.num_steps

                            % extract region around point on trajectory
                            
                            p2 = obj.im2(ppos(z, 2)-obj.curr_ps:ppos(z, 2)+obj.curr_ps, ppos(z, 1)-obj.curr_ps:ppos(z, 1)+obj.curr_ps);
                            
                            if all(p2 == p2(1))
                                p2(1, 1) = p2(1, 1) + 1;
                            end
                            
                            im2_regions(:,(z-1)*(obj.curr_ps*2+1)+1:z*(obj.curr_ps*2+1)) = single(p2);
                            
                        end                        
                        
                        height = imsz(1) + 2*obj.curr_ps;
                        width = imsz(2) + 2*obj.curr_ps;
                        
                        offsets = ppos - repmat([startx starty],obj.num_steps,1);
                        offsets = offsets';
                        
                        obj.gpu_sad2.GridSize = [ceil(height / obj.block_size_num_steps),ceil(width / obj.block_size_num_steps)];
                        
                        max(im1_padded(:))
                        min(im1_padded(:))
                        im_out_sized = feval(obj.gpu_sad2,im1_padded,im_out_sized,im2_regions,offsets(:),width,height,obj.curr_ps);
                        wait(obj.GPU)
                        max(im_out_sized(:))
                        temp = gather(im_out_sized);
                        max(temp(:))
                        xchist = temp(obj.curr_ps+1:height-obj.curr_ps,obj.curr_ps+1:width-obj.curr_ps);
                        max(xchist(:))
%                         xchist = xchist - max(xchist(:));
%                         max(xchist(:))
                        
                    else

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
                            if ~all(obj.scales == 1)
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
                        
                    end

                    if obj.random_trajectories_experimental
                        ppos = ppos_unscaled;
                        
                    end
                          
                    if obj.gpu_devel_mode
                        
                        obj.gpu_trajshift.GridSize = [ceil(imsz(1) / obj.block_size_num_steps),ceil(imsz(2) / obj.block_size_num_steps)];
                        
                        step_x = int32(ppos(:,1) - xp1);
                                                
                        step_y = int32(ppos(:,2) - yp1);
                        
                        gpu_xchist_cubed = feval(obj.gpu_trajshift,xchist_gpu.^3,gpu_xchist_cubed,step_x,step_y,imsz(2),imsz(1));
                                                
                        xcsum = gather(gpu_xchist_cubed) / obj.num_steps;
                        
                    elseif obj.gpu_processing_fail
                    
                        xcsum = xchist;
                        
                    else
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
                        
                    end
                    
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

                    fpsz = size(obj.fixedPoints);
                    
                    if obj.visuals
                        
                        % Plot figure
                        figure(fig1h);
                        
                        subplot(1,2,2);
                        
                        hold on;
                        
                        plot(obj.fixedPoints(lastnumberc:fpsz(1), 1), obj.fixedPoints(lastnumberc:fpsz(1), 2), 'rx', 'MarkerSize', 2, 'LineWidth', 1);
                        plot([obj.fixedPoints(lastnumberc:fpsz(1), 1) obj.fixedPoints(lastnumberc:fpsz(1), 1) + ppos(end,1) - xp1], [obj.fixedPoints(lastnumberc:fpsz(1), 2) obj.fixedPoints(lastnumberc:fpsz(1), 2) + ppos(end,2) - yp1], 'b-', 'MarkerSize', 5, 'LineWidth', 1);
                        
                        subplot(1,2,1);
                        
                        hold on;
                        
                        plot(obj.movingPoints(lastnumberc:fpsz(1), 1), obj.movingPoints(lastnumberc:fpsz(1), 2), 'bo', 'MarkerSize', max(1, round(strongest_local_match * 200)), 'LineWidth', 1);

%                         plot(obj.fixedPoints(lastnumberc:fpsz(1), 1), obj.fixedPoints(lastnumberc:fpsz(1), 2), 'gx', 'MarkerSize', 5, 'LineWidth', 2);
%                             plot( [obj.fixedPoints(cc, 1) obj.movingPoints(cc, 1) ], [obj.fixedPoints(cc, 2) obj.movingPoints(cc, 2)], 'r-', 'LineWidth', 1);
                        
%                         for cc = lastnumberc:fpsz(1)
% %                             plot( [obj.fixedPoints(cc, 1) obj.movingPoints(cc, 1) ], [obj.fixedPoints(cc, 2) obj.movingPoints(cc, 2)], 'r-', 'LineWidth', 1);
%                         end

                        lastnumberc = fpsz(1) + 1;
                    end

                end

            end            
            
%             obj.movingPoints(:,1) = obj.movingPoints(:,1) + obj.im1_padding(2);
%             obj.movingPoints(:,2) = obj.movingPoints(:,2) + obj.im1_padding(1);
            
%             if all(size(obj.im1_unpad_locnorm) < size(obj.im2_unpad_locnorm))
                obj.fixedPoints(:,1) = obj.fixedPoints(:,1) + obj.im2_padding(2);
                obj.fixedPoints(:,2) = obj.fixedPoints(:,2) + obj.im2_padding(1);
                obj.im2 = padarray(obj.im2,obj.im1_padding);
%             end
            
%             obj.im1 = padarray(obj.im1,obj.im1_padding);
            
            if obj.filter_points
                ind = obj.traj_strengths > median(obj.traj_strengths);
                obj.movingPoints = obj.movingPoints(ind,:);
                obj.fixedPoints = obj.fixedPoints(ind,:);
            end
            
        end % end seqslam
%%
        function cnn(obj,varargin)
        % perform image registration with cnn method

            close all

            im1_data = single(obj.im1_unpad);
            im2_data = single(obj.im2_unpad);

            if obj.test_cases(obj.curr_case).Image1.mcc
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

            end

            if obj.test_cases(obj.curr_case).Image2.mcc
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

            points1 = [];
            points2 = [];

            im1_conv1_ = im1_conv1;
            im2_conv1_ = im2_conv1;

            for i = 1:1000
                [~,m] = max(im1_conv1(:));
                [a,b] = ind2sub(size(im1_conv1),m);
                points1 = [points1; a b];
                im1_conv1(max(1,a-n1(1)):min(s1(1),a+n1(1)),max(1,b-n1(2)):min(s1(2),b+n1(2))) = 0;
                [~,m] = max(im2_conv1(:));
                [a,b] = ind2sub(size(im2_conv1),m);
                points2 = [points2; a b];
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

            % +5 due to kernel size of conv1 (will need to be changed if using
            % deeper layers)
            obj.movingPoints = validPtsOriginal(indexPairs(:,1),:) + 5;
            obj.fixedPoints = validPtsDistorted(indexPairs(:,2),:) + 5;

        end % end cnn
%%
        function ros_cnn(obj,varargin)
        % perform image registration with cnn method

            close all

            im1_data = single(obj.im1_unpad);
            im2_data = single(obj.im2_unpad);
            
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

            points1 = [];
            points2 = [];

            im1_conv1_ = im1_conv1;
            im2_conv1_ = im2_conv1;

            for i = 1:1000
                [~,m] = max(im1_conv1(:));
                [a,b] = ind2sub(size(im1_conv1),m);
                points1 = [points1; a b];
                im1_conv1(max(1,a-n1(1)):min(s1(1),a+n1(1)),max(1,b-n1(2)):min(s1(2),b+n1(2))) = 0;
                [~,m] = max(im2_conv1(:));
                [a,b] = ind2sub(size(im2_conv1),m);
                points2 = [points2; a b];
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

            % +5 due to kernel size of conv1 (will need to be changed if using
            % deeper layers)
            obj.movingPoints = validPtsOriginal(indexPairs(:,1),:) + 5;
            obj.fixedPoints = validPtsDistorted(indexPairs(:,2),:) + 5;

        end % end ros_cnn
%%
        function surf(obj,varargin)
        % perform image registration with SURF features

            % Detect features in both images.

            ptsOriginal  = detectSURFFeatures(obj.im1_fullres_gray);
            ptsDistorted = detectSURFFeatures(obj.im2_fullres_gray);

            % Extract feature descriptors.
            [featuresOriginal, validPtsOriginal] = extractFeatures(obj.im1_fullres_gray,  ptsOriginal);
            [featuresDistorted, validPtsDistorted] = extractFeatures(obj.im2_fullres_gray, ptsDistorted);

            % Match features by using their descriptors.
            indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

            % Retrieve locations of corresponding points for each image.
            obj.movingPoints = validPtsOriginal(indexPairs(:,1));
            obj.fixedPoints = validPtsDistorted(indexPairs(:,2));

        end % end surf

        function im_load(obj)
        % load and preprocess images for current case, including patch
        % normalisation if currently specified

            disp('Opening images...');
            close all

            fprintf('Case %i:\n\tContext:\t%i\n\tTranslation:\t%s\n\tLighting:\t%s\n\tScale:\t\t%s\n\tMultimodal:\t%s\n\n',obj.curr_case,obj.test_cases(obj.curr_case).context,bool2str(obj.test_cases(obj.curr_case).differences.translation),bool2str(obj.test_cases(obj.curr_case).differences.lighting),bool2str(obj.test_cases(obj.curr_case).differences.scale),bool2str(obj.test_cases(obj.curr_case).differences.multimodal));

            % load images
            obj.im1 = imread(obj.test_cases(obj.curr_case).Image1.Exposure);
            obj.im2 = imread(obj.test_cases(obj.curr_case).Image2.Exposure);

            % multimodal case, pixl images converted to 3 channel greyscale
            if obj.test_cases(obj.curr_case).differences.multimodal
                if obj.test_cases(obj.curr_case).Image1.mcc
                    obj.im1 = repmat(obj.im1,[1 1 3]);
                    obj.im1 = imresize(obj.im1,[1200 nan]);
                end
                if obj.test_cases(obj.curr_case).Image2.mcc
                    obj.im2 = repmat(obj.im2,[1 1 3]);
                    obj.im2 = imresize(obj.im2,[1200 nan]);
                end
            end

            % convert to uint8
            obj.im1 = im2uint8(obj.im1);
            obj.im2 = im2uint8(obj.im2);

            % manually correct for differing scale
            scaling = [obj.test_cases(obj.curr_case).Image1.Extension obj.test_cases(obj.curr_case).Image2.Extension];
            scaling = min(scaling) ./ scaling;
%             scaling = scaling / min(scaling);
            
            obj.scales = scaling;
            
            obj.im1_fullres_unpad = obj.im1;
            obj.im2_fullres_unpad = obj.im2;

            for i = 1:length(scaling)

%                 if scaling(i) == 1
%                     continue
%                 end

                switch i

                    case 1

                        obj.im1 = imresize(obj.im1,scaling(i));
                        obj.im1_unpad = obj.im1;
                        s1 = size(obj.im1);
                        s2 = size(obj.im2);

                        % add black padding to match other image size
                        obj.im1 = padarray(obj.im1,[max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))]);
                        
                        obj.im1_padding = [max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))];

                        obj.im1 = obj.im1(1:s2(1),1:s2(2),:);


                    case 2
                        obj.im2 = imresize(obj.im2,scaling(i));
                        obj.im2_unpad = obj.im2;
                        s1 = size(obj.im1);
                        s2 = size(obj.im2);

                        % add black padding to match other image size
                        obj.im2 = padarray(obj.im2,[max(0,round((s1(1) - s2(1))/2)), max(0,round((s1(2) - s2(2))/2))]);
                        obj.im2_padding = [max(0,round((s1(1) - s2(1))/2)), max(0,round((s1(2) - s2(2))/2))];
                        obj.im2 = obj.im2(1:s1(1),1:s1(2),:);

                end

            end

            % calculate im1 aspect ratio
            imsz = size(obj.im1);
            r = imsz(1) / imsz(2);

            % use ratio to determine image resizing dimensions
            if r > 1
                obj.yres = obj.maxdim;
                obj.xres = nan;
                padding_scale = obj.yres / imsz(1);
            else
                obj.xres = obj.maxdim;
                obj.yres = nan;
                padding_scale = obj.xres / imsz(2);
            end

            % save original full resolution image
            obj.im1_fullres = obj.im1;
            obj.im2_fullres = obj.im2;

            if length(size(obj.im1_fullres)) == 2
                obj.im1_fullres_gray = obj.im1_fullres;
            else
                obj.im1_fullres_gray = rgb2gray(obj.im1_fullres);
            end

            if length(size(obj.im2_fullres)) == 2
                obj.im2_fullres_gray = obj.im2_fullres;
            else
                obj.im2_fullres_gray = rgb2gray(obj.im2_fullres);
            end

            % resize images to maxdim 400 (one axis only)
            obj.im1_orig = imresize(obj.im1_fullres, [obj.yres obj.xres]);
            obj.im1_fullres_unpad = obj.im1_unpad;
            obj.im1_unpad = imresize(obj.im1_unpad, [scaling(1)*obj.yres scaling(1)*obj.xres]);
            obj.im1_padding = ceil(obj.im1_padding * padding_scale);
            obj.im2_orig = imresize(obj.im2_fullres, [obj.yres obj.xres]);
            obj.im2_fullres_unpad = obj.im2_unpad;
            obj.im2_unpad = imresize(obj.im2_unpad, [scaling(2)*obj.yres scaling(2)*obj.xres]);
            obj.im2_padding = ceil(obj.im2_padding * padding_scale);

            if isnan(obj.xres)
                obj.xres = size(obj.im1_orig,2);
            end
            if isnan(obj.yres)
                obj.yres = size(obj.im2_orig,1);
            end

            % get image dimensions
            i1 = size(obj.im1_orig);
            i2 = size(obj.im2_orig);

            % convert im1 to double, greyscale
            if length(i1) == 2
                obj.im1 = double(obj.im1_orig);
                obj.im1_unpad_double = double(obj.im1_unpad);
            else
                obj.im1 = double(rgb2gray(obj.im1_orig));
                obj.im1_unpad_double = double(rgb2gray(obj.im1_unpad));
            end

            % convert im2 to double, greyscale
            if length(i2) == 2
                obj.im2 = double(obj.im2_orig);
                obj.im2_unpad_double = double(obj.im2_unpad);
            else
                obj.im2 = double(rgb2gray(obj.im2_orig));
                obj.im2_unpad_double = double(rgb2gray(obj.im2_unpad));
            end

            % finished loading images
            fprintf('...done\n');

            % perform local patch normalisation
            if obj.patch_norm == 1

                disp('Patch normalising...');
                
                if obj.gpu_processing
                    
                    % Image 1

                    obj.gpu_out1 = zeros(size(obj.im1),'single','gpuArray');
                    
                    gpu_im1 = gpuArray(im2single(obj.im1));

                    [height, width] = size(obj.im1);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out1 = feval(obj.gpu_locnorm,gpu_im1,obj.gpu_out1,width,height,obj.nps,obj.minstd);

                    obj.im1 = gather(obj.gpu_out1);
                    
                    % Image 1

                    obj.gpu_out1 = zeros(size(obj.im1_unpad_double),'single','gpuArray');
                    
                    gpu_im1 = gpuArray(im2single(obj.im1_unpad_double));

                    [height, width] = size(obj.im1_unpad_double);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out1 = feval(obj.gpu_locnorm,gpu_im1,obj.gpu_out1,width,height,obj.nps,obj.minstd);

                    obj.im1_unpad_locnorm = gather(obj.gpu_out1);
                    
                    % Image 2

                    obj.gpu_out2 = zeros(size(obj.im2),'single','gpuArray');
                    
                    gpu_im2 = gpuArray(im2single(obj.im2));

                    [height, width] = size(obj.im2);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out2 = feval(obj.gpu_locnorm,gpu_im2,obj.gpu_out2,width,height,obj.nps,obj.minstd);
                    
                    obj.im2 = gather(obj.gpu_out2);
                    
                    % Image 1

                    obj.gpu_out2 = zeros(size(obj.im2_unpad_double),'single','gpuArray');
                    
                    gpu_im2 = gpuArray(im2single(obj.im2_unpad_double));

                    [height, width] = size(obj.im2_unpad_double);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out2 = feval(obj.gpu_locnorm,gpu_im2,obj.gpu_out2,width,height,obj.nps,obj.minstd);

                    obj.im2_unpad_locnorm = gather(obj.gpu_out2);
                    
                else
                    obj.im1 = locnorm(obj.im1, obj.nps, obj.minstd);
                    obj.im1_unpad_locnorm = locnorm(obj.im1_unpad_double,obj.nps,obj.minstd);
                    obj.im2 = locnorm(obj.im2, obj.nps, obj.minstd);
                    obj.im2_unpad_locnorm = locnorm(obj.im2_unpad_double,obj.nps,obj.minstd);
                end
                fprintf('...done\n');
            end

            if obj.visuals
                close all
                imshowpair(obj.im1_fullres,obj.im2_fullres,'method','montage')
%                 disp('Hit any key to continue...')
%                 pause
%                 close all
            end

        end % end im_load
%%
        function ros_imload(obj)
        % load and preprocess images for current case, including patch
        % normalisation if currently specified

            disp('Processing images...');
            close all
            % load images
            obj.curr_case = NaN;
            obj.im1 = obj.InitialImage;
            obj.im2 = obj.pixlImage;

%             mx1dim = max(size(obj.im1,2));
%             mx2dim = max(size(obj.im2,2));
            mxd = max(size(obj.im1,2),size(obj.im2,2));
            obj.im1 = imresize(obj.im1, [NaN mxd]);
            obj.im2 = imresize(obj.im2, [NaN mxd]);
            mndim = min(size(obj.im1,1),size(obj.im2,1));
            obj.im1 = obj.im1(max(1,1+floor((size(obj.im1,1) - mndim) / 2)):end-max(0,ceil((size(obj.im1,1) - mndim) / 2)),:,:);
            obj.im2 = obj.im2(max(1,1+floor((size(obj.im2,1) - mndim) / 2)):end-max(0,ceil((size(obj.im2,1) - mndim) / 2)),:,:);

            assert(all(size(obj.im1) == size(obj.im2)),'Fail!');

%             if mx1dim > mx2dim
%                 obj.im2 = imresize(obj.im2, [NaN mx1dim]);
%                 obj.im2 = padarray(obj.im2,[max(round((size(obj.im1,1)-size(obj.im2,1))/2),0), max(round((size(obj.im1,2)-size(obj.im2,2))/2),0)]); %// Change
%                 obj.im1 = padarray(obj.im1,[max(round((size(obj.im2,1)-size(obj.im1,1))/2),0), max(round((size(obj.im2,2)-size(obj.im1,2))/2),0)]); %// Change
%             else
%                 obj.im1 = imresize(obj.im1, [NaN mx2dim]);
%                 obj.im1 = padarray(obj.im1,[max(round((size(obj.im2,1)-size(obj.im1,1))/2),0), max(round((size(obj.im2,2)-size(obj.im1,2))/2),0)]); %// Change
% %                 obj.im2 = padarray(obj.im2,[max(round((size(obj.im1,1)-size(obj.im2,1))/2),0), max(round((size(obj.im1,2)-size(obj.im2,2))/2),0)]); %// Change
%                 obj.im1 = obj.im1(
%             end


            % convert to uint8
            obj.im1 = im2uint8(obj.im1);
            obj.im2 = im2uint8(obj.im2);

            % manually correct for differing scale
            scaling = obj.ros_scale;
            scaling = scaling / max(scaling);
            obj.im1_fullres_unpad = obj.im1;
            obj.im2_fullres_unpad = obj.im2;
            
            obj.scales = scaling;

            for i = 1:length(scaling)

%                 if scaling(i) == 1
%                     continue
%                 end

                switch i

                    case 1
                        obj.im1 = imresize(obj.im1,scaling(i));
                        obj.im1_unpad = obj.im1;
                        s1 = size(obj.im1);
                        s2 = size(obj.im2);

                        % add black padding to match other image size
                        obj.im1 = padarray(obj.im1,[max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))]);
                        
                        obj.im1_padding = [max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))];

                        obj.im1 = obj.im1(1:s2(1),1:s2(2),:);


                    case 2
                        obj.im2 = imresize(obj.im2,scaling(i));
                        obj.im2_unpad = obj.im2;
                        s1 = size(obj.im1);
                        s2 = size(obj.im2);

                        % add black padding to match other image size
                        obj.im2 = padarray(obj.im2,[max(0,round((s1(1) - s2(1))/2)), max(0,round((s1(2) - s2(2))/2))]);
                        
                        obj.im2_padding = [max(0,round((s1(1) - s2(1))/2)), max(0,round((s1(2) - s2(2))/2))];
                        obj.im2 = obj.im2(1:s1(1),1:s1(2),:);

                end

            end

            % calculate im1 aspect ratio
            imsz = size(obj.im1);
            r = imsz(1) / imsz(2);

            % use ratio to determine image resizing dimensions
            if r > 1
                obj.yres = obj.maxdim;
                obj.xres = nan;
                padding_scale = obj.yres / imsz(1);
            else
                obj.xres = obj.maxdim;
                obj.yres = nan;
                padding_scale = obj.xres / imsz(2);
            end

            % save original full resolution image
            obj.im1_fullres = obj.im1;
            obj.im2_fullres = obj.im2;

            if length(size(obj.im1_fullres)) == 2
                obj.im1_fullres_gray = obj.im1_fullres;
            else
                
            obj.im1_fullres_gray = rgb2gray(obj.im1_fullres);
                
            end

            if length(size(obj.im2_fullres)) == 2
                obj.im2_fullres_gray = obj.im2_fullres;
            else
                
            obj.im2_fullres_gray = rgb2gray(obj.im2_fullres);
              
            end

            % resize images to maxdim 400 (one axis only)
            obj.im1_orig = imresize(obj.im1_fullres, [obj.yres obj.xres]);
            obj.im1_fullres_unpad = obj.im1_unpad;
            obj.im1_unpad = imresize(obj.im1_unpad, [scaling(1)*obj.yres scaling(1)*obj.xres]);
            obj.im1_padding = ceil(obj.im1_padding * padding_scale);
            obj.im2_orig = imresize(obj.im2_fullres, [obj.yres obj.xres]);
            obj.im2_fullres_unpad = obj.im2_unpad;
            obj.im2_unpad = imresize(obj.im2_unpad, [scaling(2)*obj.yres scaling(2)*obj.xres]);
            obj.im2_padding = ceil(obj.im2_padding * padding_scale);

            if isnan(obj.xres)
                obj.xres = size(obj.im1_orig,2);
            end
            if isnan(obj.yres)
                obj.yres = size(obj.im2_orig,1);
            end

            % get image dimensions
            i1 = size(obj.im1_orig);
            i2 = size(obj.im2_orig);

            % convert im1 to double, greyscale
            if length(i1) == 2
                obj.im1 = double(obj.im1_orig);
                obj.im1_unpad_double = double(obj.im1_unpad);
            else
                obj.im1 = double(rgb2gray(obj.im1_orig));
                obj.im1_unpad_double = double(rgb2gray(obj.im1_unpad));
            end

            % convert im2 to double, greyscale
            if length(i2) == 2
                obj.im2 = double(obj.im2_orig);
            else
                obj.im2 = double(rgb2gray(obj.im2_orig));
            end

            % finished loading images
            fprintf('...done\n');

            % perform local patch normalisation
            if obj.patch_norm == 1

                disp('Patch normalising...');
                
                if obj.gpu_processing
                    
                    % Image 1

                    obj.gpu_out1 = zeros(size(obj.im1),'single','gpuArray');
                    
                    gpu_im1 = gpuArray(im2single(obj.im1));

                    [height, width] = size(obj.im1);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out1 = feval(obj.gpu_locnorm,gpu_im1,obj.gpu_out1,width,height,obj.nps,obj.minstd);

                    obj.im1 = gather(obj.gpu_out1);
                    
                    % Image 1

                    obj.gpu_out1 = zeros(size(obj.im1_unpad_double),'single','gpuArray');
                    
                    gpu_im1 = gpuArray(im2single(obj.im1_unpad_double));

                    [height, width] = size(obj.im1_unpad_double);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out1 = feval(obj.gpu_locnorm,gpu_im1,obj.gpu_out1,width,height,obj.nps,obj.minstd);

                    obj.im1_unpad_locnorm = gather(obj.gpu_out1);
                    
                    % Image 2

                    obj.gpu_out2 = zeros(size(obj.im2),'single','gpuArray');
                    
                    gpu_im2 = gpuArray(im2single(obj.im2));

                    [height, width] = size(obj.im2);

                    obj.gpu_locnorm.GridSize = [ceil(height / obj.block_size1),ceil(width / obj.block_size1),1];

                    obj.gpu_out2 = feval(obj.gpu_locnorm,gpu_im2,obj.gpu_out2,width,height,obj.nps,obj.minstd);
                    
                    obj.im2 = gather(obj.gpu_out2);
                    
                else
                    
                    obj.im1 = locnorm(obj.im1,obj.nps,obj.minstd);
                    obj.im1_unpad_locnorm = locnorm(obj.im1_unpad_double,obj.nps,obj.minstd);
                    obj.im2 = locnorm(obj.im2,obj.nps,obj.minstd);
                    
                end
                
                fprintf('...done\n');
            end
            
            if obj.visuals
                close all
                imshowpair(obj.im1_fullres,obj.im2_fullres,'method','montage')
                figure
                imshowpair(obj.im1,obj.im2,'method','montage')
            end

        end % end im_load
%%
        function align_im(obj)
        % align images for current case

            close all;

            maxdistancefract = 0.01;
            mp = obj.movingPoints;
            fp = obj.fixedPoints;

            if strcmp(obj.method,'cnn')
                im1f = obj.im1_fullres_unpad;
                im2f = obj.im2_fullres_unpad;
                im1s = obj.im1_unpad;
                im2s = obj.im2_unpad;
            else
                im1f = obj.im1_fullres;
                im2f = obj.im2_fullres;
                im1s = obj.im1_orig;
                im2s = obj.im2_orig;
            end

            % get dimensions of full-res images
            im1fullsz = size(im1f);
            im2fullsz = size(im2f);


            % get dimensions of lo-res images
            xres1 = size(im1s,2);
            yres1 = size(im1s,1);
            xres2 = size(im2s,2);
            yres2 = size(im2s,1);

            if ~strcmp(obj.method,'surf')
                mp(:, 1) = mp(:, 1) * (im2fullsz(2) / xres2);
                mp(:, 2) = mp(:, 2) * (im2fullsz(1) / yres2);
                fp(:, 1) = fp(:, 1) * (im1fullsz(2) / xres1);
                fp(:, 2) = fp(:, 2) * (im1fullsz(1) / yres1);
            end

            % find the maximum dimension of the two images
            maxdim = max([im1fullsz(1:2) im2fullsz(1:2)]);

            [obj.tform,inlierPtsOut,inlierPtsIn,obj.tform_status] = estimateGeometricTransform(mp, fp,'similarity','MaxDistance', maxdistancefract * maxdim);
            obj.matchedOriginal = inlierPtsOut;
            obj.matchedDistorted = inlierPtsIn;
            
            if obj.tform_status == 0

                % calculate some output values
                Tinv  = obj.tform.invert.T;
                ss = Tinv(2,1);
                sc = Tinv(1,1);
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]) = struct();
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered = sqrt(ss*ss + sc*sc);
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).thetaRecovered = atan2(ss,sc)*180/pi;
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).percentPtsUsed = size(inlierPtsOut,1)/size(mp,1);
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform = obj.tform;
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform_status = obj.tform_status;
            
                if obj.test_points_used
                    obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).fraction_pts_used = size(inlierPtsIn,1) / size(mp,1);
                    obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).min_pts_used = size(inlierPtsIn,1) >= obj.min_pts_required;
                end

                obj.test_match();
                if obj.test_points_used
                    fprintf('Case %i:\n\tMethod: %s\n\tScale: %f\n\tTheta: %f\n\t%s Pts Used: %0.1f%s\n\tPoints Used: %s\n\tFraction Points: %0.3f\n\tMessage: %s\n\n',obj.curr_case,obj.method,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).thetaRecovered,'%',obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).percentPtsUsed*100,'%',bool2str(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).min_pts_used),obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).fraction_pts_used,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).message);
                else
                    fprintf('Case %i:\n\tMethod: %s\n\tScale: %f\n\tTheta: %f\n\t%s Pts Used: %0.1f%s\n\tMessage: %s\n\n',obj.curr_case,obj.method,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).thetaRecovered,'%',obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).percentPtsUsed*100,'%',obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).message);
                end
            else
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform_status = obj.tform_status;
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).match = 0;
                if obj.tform_status == 1
                    s = 'Not enough matched points.';
                else
                    s = 'Not enough inliers have been found.';
                end
                fprintf('Case %i:\n\tMethod: %s\n\t%s\n',obj.curr_case,obj.method,s);
            end

%             obj.save_prog();

        end % end align_im

%%
        function ros_align_im(obj)
        % align images for current case

            close all;

            maxdistancefract = 0.01;
            mp = obj.movingPoints;
            fp = obj.fixedPoints;

            if strcmp(obj.method,'cnn')
                im1f = obj.im1_fullres_unpad;
                im2f = obj.im2_fullres_unpad;
                im1s = obj.im1_unpad;
                im2s = obj.im2_unpad;
            else
                im1f = obj.im1_fullres;
                im2f = obj.im2_fullres;
                im1s = obj.im1_orig;
                im2s = obj.im2_orig;
            end

            % get dimensions of full-res images
            im1fullsz = size(im1f);
            im2fullsz = size(im2f);


            % get dimensions of lo-res images
            xres1 = size(im1s,2);
            yres1 = size(im1s,1);
            xres2 = size(im2s,2);
            yres2 = size(im2s,1);

            if ~strcmp(obj.method,'surf') && ~strcmp(obj.method,'ros_surf')
                mp(:, 1) = mp(:, 1) * (im2fullsz(2) / xres2);
                mp(:, 2) = mp(:, 2) * (im2fullsz(1) / yres2);
                fp(:, 1) = fp(:, 1) * (im1fullsz(2) / xres1);
                fp(:, 2) = fp(:, 2) * (im1fullsz(1) / yres1);
            end

            % find the maximum dimension of the two images
            maxdim = max([im1fullsz(1:2) im2fullsz(1:2)]);

            [obj.tform,inlierPtsOut,inlierPtsIn,obj.tform_status] = estimateGeometricTransform(mp, fp,'similarity','MaxDistance', maxdistancefract * maxdim);
            obj.matchedOriginal = inlierPtsOut;
            obj.matchedDistorted = inlierPtsIn;
            
            if obj.test_points_used
                obj.fraction_pts_used = size(inlierPtsIn,1) / size(mp,1);
                obj.min_pts_used = size(inlierPtsIn,1) >= obj.min_pts_required;
            end

        end % end ros_align_im
%%
        function align_im2(obj)
        % align images for current case

            close all;

            maxdistancefract = 0.01;

            mp = obj.movingPoints;
            fp = obj.fixedPoints;

            % get dimensions of full-res images
            im1fullsz = size(obj.im1_fullres);
            im2fullsz = size(obj.im2_fullres);
            if strcmp(obj.method,'cnn')
                im1fullsz = size(obj.im1_fullres_unpad);
                im2fullsz = size(obj.im2_fullres_unpad);
            end


            % get dimensions of lo-res images
            assert(size(obj.im1_orig,2) == size(obj.im2_orig,2),'im1_orig and im2_orig second dimension not equal');
            assert(size(obj.im1_orig,1) == size(obj.im2_orig,1),'im1_orig and im2_orig first dimension not equal');
            xres1 = size(obj.im1_orig,2);
            yres1 = size(obj.im1_orig,1);
            xres2 = size(obj.im1_orig,2);
            yres2 = size(obj.im1_orig,1);

            if strcmp(obj.method,'cnn')
                xres1 = size(obj.im1_unpad,2);
                yres1 = size(obj.im1_unpad,1);
                xres2 = size(obj.im2_unpad,2);
                yres2 = size(obj.im2_unpad,1);
            end

            if strcmp(obj.method,'seqslam') || strcmp(obj.method,'new_seqslam')
                % re-map the moving and fixed points to the full scale images
                mp(:, 1) = mp(:, 1) * (im2fullsz(2) / xres2);
                mp(:, 2) = mp(:, 2) * (im2fullsz(1) / yres2);
                fp(:, 1) = fp(:, 1) * (im1fullsz(2) / xres1);
                fp(:, 2) = fp(:, 2) * (im1fullsz(1) / yres1);
            end

            if strcmp(obj.method,'cnn')
                % re-map the moving and fixed points to the full scale images
                fp(:, 1) = fp(:, 1) * (im2fullsz(2) / xres2);
                fp(:, 2) = fp(:, 2) * (im2fullsz(1) / yres2);
                mp(:, 1) = mp(:, 1) * (im1fullsz(2) / xres1);
                mp(:, 2) = mp(:, 2) * (im1fullsz(1) / yres1);
            end

            % find the maximum dimension of the two images
            maxdim = max([im1fullsz(1:2) im2fullsz(1:2)]);

            [obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform,inlierPtsOut,inlierPtsIn,obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform_status] = estimateGeometricTransform(mp, fp,'similarity','MaxDistance', maxdistancefract * maxdim);
            obj.matchedOriginal = inlierPtsOut;
            obj.matchedDistorted = inlierPtsIn;

            % check if tform found and reasonable
            if obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform_status ~= 0
                fprintf('Transform for case %i not found!\n',obj.curr_case)
                return
            elseif obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered > 2
                fprintf('Scale for case %i too large!\n',obj.curr_case)
                return
            elseif obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered < 0.5
                fprintf('Scale for case %i too small!\n',obj.curr_case)
                return
            elseif abs(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).thetaRecovered) > 7.5
                fprintf('Rotation for case %i too large!\n',obj.curr_case)
                return
            end

            % create number of frames, desired dimension
            nframes = 10;
            desdim = 800;

            if ~strcmp(obj.method,'cnn')
                [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(obj.im2_fullres,2)], [1 size(obj.im2_fullres,1)]);

                % Find the minimum and maximum output limits
                xMin = min([1; xLimitsOut(:)]);
                xMax = max([size(obj.im1_fullres,2); xLimitsOut(:)]);

                yMin = min([1; yLimitsOut(:)]);
                yMax = max([size(obj.im1_fullres,1); yLimitsOut(:)]);

                % Width and height of outView.
                width  = round(xMax - xMin);
                height = round(yMax - yMin);

                xLimits = [xMin xMax];
                yLimits = [yMin yMax];
                outView = imref2d([height width], xLimits, yLimits);

                im2_registered = imwarp(obj.im2_fullres, projective2d(eye(3)), 'OutputView', outView);

                registered = imwarp(obj.im1_fullres, obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, 'OutputView', outView);%imref2d(size(obj.im2_fullres)));
            else
                [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(obj.im2_unpad,2)], [1 size(obj.im2_unpad,1)]);

                % Find the minimum and maximum output limits
                xMin = min([1; xLimitsOut(:)]);
                xMax = max([size(obj.im1_unpad,2); xLimitsOut(:)]);

                yMin = min([1; yLimitsOut(:)]);
                yMax = max([size(obj.im1_unpad,1); yLimitsOut(:)]);

                % Width and height of outView.
                width  = round(xMax - xMin);
                height = round(yMax - yMin);

                xLimits = [xMin xMax];
                yLimits = [yMin yMax];
                outView = imref2d([height width], xLimits, yLimits);

                im2_registered = imwarp(obj.im2_unpad, projective2d(eye(3)), 'OutputView', outView);

                registered = imwarp(obj.im1_unpad, obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, 'OutputView', outView);%imref2d(size(obj.im2_fullres)));
            end

            obj.im1_reg = registered;
            obj.im2_reg = im2_registered;

        end
%%
        % set the path to the network definition file
        function set_network(obj,network)

            if ~strcmp('.prototxt',network(end-8:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.network = network;
            obj.load_model();

        end % end set_network
%%
        % set the path to the network definition file
        function set_network_col(obj,network)

            if ~strcmp('.prototxt',network(end-8:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.network_col = network;
            obj.load_model();

        end % end set_network
%%
        % set the path to the weights file
        function set_weights(obj,weights)

            if ~strcmp('.caffemodel',weights(end-10:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.weights = weights;
            obj.load_model();

        end % end set_weights
%%
        % set the path to the weights file
        function set_weights_col(obj,weights)

            if ~strcmp('.caffemodel',weights(end-10:end))
                disp('Invalid filetype. Please use a caffe prototxt network definition file.')
                return
            end

            obj.weights_col = weights;
            obj.load_model();

        end % end set_weights
%%
        % set the caffe model as a preloaded model
        function set_model(obj,model)

            obj.model = model;

        end % end set_model
%%
%%
        function load_model(obj)

            if ~isempty(obj.weights) && ~isempty(obj.network)

                disp('Loading caffe model...')

                try
                    obj.model = caffe.Net(obj.network,obj.weights,'test');
                catch
                    disp('Model not loaded.')
                end

            end

            if ~isempty(obj.weights_col) && ~isempty(obj.network_col)

                disp('Loading caffe model...')

                try
                    obj.model_col = caffe.Net(obj.network_col,obj.weights_col,'test');
                catch
                    disp('Model not loaded.')
                end

            end

        end % end load_model
%%
        function bad_matches(obj)

            for i = 1:length(obj.known_bad_cases)

                methods = {'seqslam','cnn','surf'};

                for j = 1:size(methods,1)
                    obj.known_bad_cases(i), methods(j)
                    obj.test_cases(obj.known_bad_cases(i)).(methods(j)).match = 0;

                end


            end

        end
%%
        function [match,message] = test_match(obj)

            match = 1;

            message = '';

            if any(obj.known_bad_cases == obj.curr_case)

                match = 0;

                message = strcat(message,'Bad case. ');

            end

            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'tform_status')

                if obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform_status ~= 0

                    message = strcat(message,'tform_status not 0. ');

                    match = 0;

                end

            else

                message = strcat(message,'tform_status missing. ');

                match = 0;

            end

            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'scaleRecovered')

                if obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered < 0.5 || obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).scaleRecovered > 5

                    message = strcat(message,'Scale. ');

                    match = 0;

                end

            end

            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'thetaRecovered')

                if abs(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).thetaRecovered) > 15

                    message = strcat(message,'Rotation. ');

                    match = 0;

                end

            end
            
            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'fraction_pts_used')

                if obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).fraction_pts_used < obj.min_fraction_pts_used

                    message = strcat(message,'Min fraction points for transform estimation not met. ');

                    match = 0;
                end
            end
            
            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'min_pts_used')

                if ~obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).min_pts_used

                    message = strcat(message,'Min number points for transform estimation not met. ');

                    match = 0;
                end
                
            end

            if isfield(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]), 'tform')

                im1sz = size(obj.im1_fullres_unpad);
                im2sz = size(obj.im2_fullres_unpad);

                imsz = [max(im1sz(1), im2sz(1)) max(im1sz(2), im2sz(2))];

                if strcmp(obj.method,'cnn')
                    v = (im1sz(1) - im2sz(1)) /2;
                    u = (im1sz(2) - im2sz(2)) /2;
                else
                    v = 0;
                    u = 0;
                end

                diff_X = obj.test_cases(obj.curr_case).Image1.Translation_X - obj.test_cases(obj.curr_case).Image2.Translation_X;
                diff_Y = obj.test_cases(obj.curr_case).Image1.Translation_Y - obj.test_cases(obj.curr_case).Image2.Translation_Y;


                if abs(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform.T(3,2) + v + diff_Y*im1sz(1)/2) > obj.trans_limit * imsz(1) || abs(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform.T(3,1) + u + diff_X*im1sz(2)/2) > obj.trans_limit * imsz(2)

                    message = strcat(message,'Translation. ');

                    match = 0;
                end

            end

            obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).match = match;
            obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).message = message;

        end

    end

end

function string = bool2str(b)

string = regexprep(sprintf('%i',boolean(b)),{'1','0'},{'True','False'});

end