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
        network = '/home/james/ros_ws/src/seqreg_tpp/matlab/caffe/deploy_conv1.prototxt'
        network_col = '/home/james/ros_ws/src/seqreg_tpp/matlab/caffe/deploy_conv1_3.prototxt'
        model = '/home/james/ros_ws/src/seqreg_tpp/matlab/caffe/imagenet_grey_weights.caffemodel'
        model_col = '/home/james/ros_ws/src/seqreg_tpp/matlab/caffe/bvlc_reference_caffenet.caffemodel'
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
        seqregSrv
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
        curr_ps
        ros_initialised
        results_publisher
        kernel_path = '/home/james/co/SeqSLAM_GPU';
        registrator
        image_pair
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
        %   seqreg     An empty structure for storing SeqSLAM results
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
                        
            obj.results_publisher = ResultsPublisher(obj);
            
            obj.registrator = ImageRegistration();
            
            obj.init_ros();                       

        end % end init
%%
        function init_ros(obj)
            
            if obj.ros_initialised
                rosshutdown;                
            end
            
            rosinit;
            obj.ros_initialised = true;            
            obj.seqregSrv = rossvcserver('/seqreg_tpp/seqreg','seqreg_tpp/MATLABSrv',@obj.seqregCallback);
            
        end
        
%%
        % accepts custom srv type
        function res = seqregCallback(obj,~,req,res)
            
            fprintf('%s started\n',req.Method.Data);

            [initial_image,~] = readImage(req.InitialImage);
            
            [secondary_image,~] = readImage(req.SecondaryImage);
            
%             obj.save_string = num2str(req.SecondaryImage.Header.Stamp.Sec);
                        
            obj.image_pair = ImagePair(initial_image,secondary_image,req.Scales.Data');
            
%             obj.curr_trial = obj.curr_trial + 1;
            
            fprintf('Initial registration with SeqSLAM to estimate new pose\n');
            
            if obj.visuals                
                close all
                imshowpair(obj.image_pair.im1_orig,obj.image_pair.im2_orig,'method','montage');
                pause
                close all 
            end
            
            % perform image registration, returns ImageRegistrationResult
            % object
            obj.results = obj.registrator.process(req.Method.Data,obj.image_pair);
                        
%             obj.ros_save_im();
            
            if obj.results.registration_successful

                fprintf('Initial Registration\n\tEstimated Scale:\t%0.3f\n\tEstimated Rotation:\t%0.3f\n\tTrans X: %0.3f\n\tTrans Y: %0.3f\n\tPoints Used: %s\n\tFraction Points: %0.3f\n',obj.results.scaleRecovered,obj.results.thetaRecovered,obj.results.tform.T(3,2),obj.results.tform.T(3,1),bool2str(obj.results.min_pts_used),obj.results.fraction_pts_used);

                if abs(1 - obj.results.scaleRecovered) < 2.0 && abs(obj.results.thetaRecovered) < 10 && obj.results.percentPtsUsed > obj.min_fraction_pts_used && obj.results.min_pts_used
                    res.Results.Data = [scaleRecovered thetaRecovered obj.tform.T(3,2) obj.tform.T(3,1)];
                    res.Success.Data = true;
                    res.Message.Data = ['Registration with ' req.Method.Data ' succeeded!'];
                    fprintf(['Registration with ' req.Method.Data ' succeeded!\n']);
                else
                    res.Results.Data = [0.0 0.0 0.0 0.0];
                    res.Success.Data = false;
                    res.Message.Data = ['Registration with ' req.Method.Data ' didnt meet limits!'];
                    fprintf(['Registration with ' req.Method.Data ' didnt meet limits!\n']);
                end
            else
                res.Results.Data = [0.0 0.0 0.0 0.0];
                res.Success.Data = false;
                res.Message.Data = ['Registration with ' req.Method.Data ' failed!'];
                fprintf(['Registration with ' req.Method.Data ' failed!\n']);
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
            
            obj.image_pair.toggle_vis();
            obj.registrator.toggle_vis();

        end % end toggle_vis
        
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
        
        function [match,message] = test_match(obj)
            match = true;
            message = '';
            
            if ~obj.results.registration_successful

                message = strcat(message,'Registration failed. ');

                match = false;
                
            end 

            if ~isempty(obj.results.tform_status)

                if obj.results.tform_status ~= 0

                    message = strcat(message,'tform_status not 0. ');

                    match = false;

                end

            else

                message = strcat(message,'tform_status missing. ');

                match = false;

            end

            if ~isempty(obj.results.scaleRecovered)

                if obj.results.scaleRecovered < 0.5 || obj.results.scaleRecovered > 5

                    message = strcat(message,'Scale. ');

                    match = false;

                end

            end

            if ~isempty(obj.results.thetaRecovered)

                if abs(obj.results.thetaRecovered) > 15

                    message = strcat(message,'Rotation. ');

                    match = false;

                end

            end

%             if ~isempty(obj.results.fraction_pts_used)
% 
%                 if obj.results.fraction_pts_used < obj.min_fraction_pts_used
% 
%                     message = strcat(message,'Min fraction points for transform estimation not met. ');
% 
%                     match = false;
%                 end
%             end
% 
%             if ~isempty(obj.results.min_pts_used)
% 
%                 if ~obj.results.min_pts_used
% 
%                     message = strcat(message,'Min number points for transform estimation not met. ');
% 
%                     match = false;
%                 end
% 
%             end
            
            if ~isempty(obj.results.tform)

                im1sz = size(obj.im1);
                im2sz = size(obj.im2);

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


                if abs(obj.results.tform.T(3,2) + v + diff_Y*im1sz(1)/2) > obj.trans_limit * imsz(1) || abs(obj.results.tform.T(3,1) + u + diff_X*im1sz(2)/2) > obj.trans_limit * imsz(2)

                    message = strcat(message,'Translation. ');

                    match = false;
                end

            end

            obj.results.match = match;
            obj.results.message = message;

        end
        
        function save_prog(obj)
        % save the object to backup results

            d = clock;
            nasa = obj;
            save(sprintf('%snasa-%i%02i%02i.mat',obj.save_dir,d(1),d(2),d(3)),'nasa');

        end
        
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
                if strcmp(obj.method,'seqreg')
                    counts(ind) = counts(ind) + obj.test_cases(i).([obj.method num2str(obj.trajectory_mode)]).match;
                else
                    counts(ind) = counts(ind) + obj.test_cases(i).(obj.method).match;
                end
            end

            obj.results.([obj.method num2str(obj.trajectory_mode)]) = counts;

            obj.test_totals = total_counts;

        end % end summ_testing
        
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
                
                disp(['Processing case ' num2str(i)]);
                obj.curr_case = i;

                im1 = imread(obj.test_cases(obj.curr_case).Image1.Exposure);
                im2 = imread(obj.test_cases(obj.curr_case).Image2.Exposure);
                scales = 1./[obj.test_cases(obj.curr_case).Image2.Extension obj.test_cases(obj.curr_case).Image1.Extension];
                
                obj.image_pair = ImagePair(im1,im2,scales);
                
%                 tic
%                 obj.method = 'cnn';
%                 obj.results = obj.registrator.process(obj.method,obj.image_pair);
%                 obj.im1 = obj.image_pair.im1_orig;
%                 obj.im2 = obj.image_pair.im2_orig;
%                 obj.test_match();
%                 obj.test_cases(obj.curr_case).(obj.method) = struct(obj.results);
%                 obj.save_im();
%                 fprintf('CNN Time: %0.4f\n',toc)
                
                tic
                obj.method = 'surf';
                obj.results = obj.registrator.process(obj.method,obj.image_pair);
                obj.im1 = obj.image_pair.im1_fullres_gray;
                obj.im2 = obj.image_pair.im2_fullres_gray;
                obj.test_match();
                warning off
                obj.test_cases(obj.curr_case).(obj.method) = struct(obj.results);
                warning on
%                 obj.save_im();
                
                fprintf('SURF Time: %0.4f\n',toc)
                
                tic
                obj.method = 'seqreg';
                obj.registrator.set_trajectory_mode(obj.trajectory_mode);
                obj.results = obj.registrator.process(obj.method,obj.image_pair);
                obj.im1 = obj.image_pair.im1;
                obj.im2 = obj.image_pair.im2;
                obj.test_match();
                warning off
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]) = struct(obj.results);
                warning on
%                 obj.save_im();
                fprintf('SeqReg Time: %0.4f\n',toc)
                
%                 obj.save_prog();

            end

        end

    end

end