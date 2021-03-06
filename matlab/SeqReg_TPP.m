classdef SeqReg_TPP < handle
    % SEQREG_TPP Summary of this class goes here
    %   Detailed explanation goes here
%%
    properties
        known_bad_cases
        test_cases
        curr_case
        method
        visuals = false
        save_dir = '/home/james/testing'
        gif_dir = '/home/james/datasets/pixl-watson/gifs/'
        dataset_dir = '/home/james/datasets/pixl-watson'
        results
        seqregSrv
        save_images = false
        trajectory_mode = 0
        min_pts_required = 4;
        min_fraction_pts_used = 0.1
        ros_initialised
        results_publisher
        registrator
        image_pair
        save_gif = true;
        open_gif = false;
        parameters
        ros_results
    end
%%
    methods

%%
        function obj = SeqReg_TPP(varargin)
        % SeqReg_TPP constructor

            obj.init();

        end % end SeqReg_TPP
        
        function init(obj)
        % initialise the object, can be used to reinitialise when reloading
            close all
            rng('shuffle');
                        
            obj.results_publisher = ResultsPublisher(obj);
            
            obj.test_cases = load([fileparts(mfilename('fullpath')) '/test_cases.mat'], 'test_cases');
            obj.parameters = load([fileparts(mfilename('fullpath')) '/parameters.mat'], 'parameters');
            obj.parameters = obj.parameters.parameters;
            
            in_args = struct2optarg(obj.parameters.image_registration);
            obj.registrator = ImageRegistration(in_args{:});
            
            obj.init_ros(); 
            
            in_args = struct2optarg(obj.parameters.image_pair);
            obj.image_pair = ImagePair(in_args{:});

        end % end init
        
        function init_ros(obj)
            
            if obj.ros_initialised
                rosshutdown;                
            end
            
            rosinit;
            obj.ros_initialised = true;            
            
            obj.seqregSrv = rossvcserver('/seqreg_tpp/seqreg','user_input/MATLABSrv',@obj.seqregCallback);
                        
        end
        
        % accepts custom srv type
        function res = seqregCallback(obj,~,req,res)
            
            warning('off','MATLAB:structOnObject');
            
            fprintf('Service call received for %s\n',req.Method.Data);

            [initial_image,~] = readImage(req.InitialImage);
            
            [secondary_image,~] = readImage(req.SecondaryImage);
            
            req.Scales.Data
                                    
            obj.image_pair.set_images(initial_image,secondary_image,req.Scales.Data');
            
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
            
            if obj.results.registration_successful
            
                if obj.save_images
                    obj.warp_im();
                    obj.generate_gif(obj.gif_dir, ['ros_' obj.method num2str(obj.trajectory_mode)], ['ros_' obj.method '_' obj.curr_case]);
                end

                fprintf('Initial Registration\n\tEstimated Scale:\t%0.3f\n\tEstimated Rotation:\t%0.3f\n\tTrans X: %0.3f\n\tTrans Y: %0.3f\n\tPoints Used: %s\n\tFraction Points: %0.3f\n',obj.results.scaleRecovered,obj.results.thetaRecovered,obj.results.tform.T(3,2),obj.results.tform.T(3,1),bool2str(obj.results.min_pts_used),obj.results.percentPtsUsed);

                if abs(1 - obj.results.scaleRecovered) < obj.parameters.ros.scale_tolerance && abs(obj.results.thetaRecovered) < obj.parameters.ros.rotation_tolerance %&& obj.results.percentPtsUsed > obj.min_fraction_pts_used && obj.results.min_pts_used
                    res.Results.Data = [obj.results.scaleRecovered obj.results.thetaRecovered obj.results.tform.T(3,2) obj.results.tform.T(3,1)];
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
                
                imshowpair(obj.results.im1_registered,obj.results.im2_registered)
            end
            
            result_struct = struct('request', req, 'response', res, 'results', struct(obj.results));
            
            if isempty(obj.ros_results)
                obj.ros_results = result_struct;                
            else
                obj.ros_results(end + 1) = result_struct;
            end
            
            if mod(length(obj.ros_results),obj.parameters.ros.save_frequency) == 0
                obj.save_prog();
            end

        end
        
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
        
        % Dataset only: test if alignment is correct
        function [match,message] = test_match(obj,im1,im2)
            
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

                if obj.results.scaleRecovered < obj.parameters.dataset.min_scale || obj.results.scaleRecovered > obj.parameters.dataset.max_scale

                    message = strcat(message,'Scale. ');

                    match = false;

                end

            end

            if ~isempty(obj.results.thetaRecovered)

                if abs(obj.results.thetaRecovered) > obj.parameters.dataset.rotation_tolerance

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

                im1sz = size(im1);
                im2sz = size(im2);

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

                if abs(obj.results.tform.T(3,2) + v + diff_Y*im1sz(1)/2) > obj.parameters.dataset.translation_limit * imsz(1) || abs(obj.results.tform.T(3,1) + u + diff_X*im1sz(2)/2) > obj.parameters.dataset.translation_limit * imsz(2)

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
            seqreg_tpp = obj; %#ok<NASGU>
            save(sprintf('%s/seqreg_tpp-%i%02i%02i.mat',obj.save_dir,d(1),d(2),d(3)),'seqreg_tpp');

        end
        
        function test_dataset(obj,varargin)
        % test all or some cases using all methods
            warning('off','MATLAB:structOnObject');
            warning('off','robotics:ros:common:SavedObjectInvalid');
            
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
                
                fprintf(['\nProcessing case ' num2str(i) '\n']);
                obj.curr_case = i;
                
                im1_current = imread([obj.dataset_dir '/' obj.test_cases(obj.curr_case).Image1.Exposure]);
                im2_current = imread([obj.dataset_dir '/' obj.test_cases(obj.curr_case).Image2.Exposure]);
                scales_current = 1./[obj.test_cases(obj.curr_case).Image2.Extension obj.test_cases(obj.curr_case).Image1.Extension];
                
                obj.image_pair.set_images(im1_current,im2_current,scales_current);
                
                tic
                obj.method = 'cnn';
                obj.results = obj.registrator.process(obj.method,obj.image_pair);
                fprintf('CNN Time: %0.4f\n',toc)
                obj.test_match(obj.image_pair.im1_orig,obj.image_pair.im2_orig);
                obj.test_cases(obj.curr_case).(obj.method) = struct(obj.results);            
                if obj.save_images && obj.results.match
                    obj.warp_im();
                    obj.generate_gif(obj.gif_dir, obj.method, [obj.method '_' obj.curr_case]);
                end
                
                tic
                obj.method = 'surf';
                obj.results = obj.registrator.process(obj.method,obj.image_pair);
                fprintf('SURF Time: %0.4f\n',toc)
                obj.test_match(obj.image_pair.im1_fullres_gray,obj.image_pair.im2_fullres_gray);
                obj.test_cases(obj.curr_case).(obj.method) = struct(obj.results); 
                if obj.save_images && obj.results.match
                    obj.warp_im();
                    obj.generate_gif(obj.gif_dir, obj.method, [obj.method '_' obj.curr_case]);
                end                
                
                tic
                obj.method = 'seqreg';
                obj.registrator.set_trajectory_mode(obj.trajectory_mode);
                obj.results = obj.registrator.process(obj.method,obj.image_pair);
                fprintf('SeqReg Time: %0.4f\n',toc)
                obj.test_match(obj.image_pair.im1,obj.image_pair.im2);
                obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]) = struct(obj.results);  
                if obj.save_images && obj.results.match
                    obj.warp_im();
                    obj.generate_gif(obj.gif_dir, [obj.method num2str(obj.trajectory_mode)], [obj.method '_' obj.curr_case]);
                end                
                
                if mod(i,obj.parameters.dataset.save_frequency) == 0
                    obj.save_prog();
                end
                
%                 obj.results_publisher.generate_html();
%                 
%                 if mod(i,obj.parameters.dataset.results_frequency) == 0
%                     obj.results_publisher.update_home();
%                     obj.results_publisher.generate_fullresultshtml();
%                 end

            end

        end
        
        

        function warp_im(obj)
        % warp images based on returned registration transform

            % depending on method, use padded or unpadded image
            if strcmp(obj.method,'cnn')
                im1f = obj.image_pair.im1_fullres_unpad;
                im2f = obj.image_pair.im2_fullres_unpad;
            else
                im1f = obj.image_pair.im1_fullres;
                im2f = obj.image_pair.im2_fullres;
            end

            % create number of frames, desired dimension
            desdim = 800;

            % different methods use different fixed and moving images,
            % should update this to make consistent
            if ~strcmp(obj.method,'cnn')
                [xLimitsOut,yLimitsOut] = outputLimits(obj.results.tform, [1 size(im2f,2)], [1 size(im2f,1)]);

                % Find the minimum and maximum output limits
                xMin = min([1; xLimitsOut(:)]);
                xMax = max([size(im1f,2); xLimitsOut(:)]);
                yMin = min([1; yLimitsOut(:)]);
                yMax = max([size(im1f,1); yLimitsOut(:)]);

            else
                [xLimitsOut,yLimitsOut] = outputLimits(obj.results.tform, [1 size(im1f,2)], [1 size(im1f,1)]);

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
            im2_registered = imwarp(im2f, affine2d(eye(3)), 'OutputView', outView);

            % apply estimated transform to im1
            registered = imwarp(im1f, obj.results.tform, 'OutputView', outView);

            % remove any excess padding from both images
            mask = im2_registered == 0 & registered == 0;
            mask = prod(mask,3);
            mask_test = double(repmat(all(mask'),[size(mask,2),1]))' | double(repmat(all(mask),[size(mask,1),1]));
            im2_registered = im2_registered(~all(mask_test'),~all(mask_test),:);
            registered = registered(~all(mask_test'),~all(mask_test),:);

            % resize to max dimension of desdim
            max_dim = max(size(registered));
            r_dim = [NaN NaN];
            r_dim(size(registered) == max_dim) = desdim;
            im1_registered = imresize(registered,r_dim);
            im2_registered = imresize(im2_registered,r_dim);
            
            obj.results.im1_registered = im1_registered;
            obj.results.im2_registered = im2_registered;

        end
        
        function generate_gif(obj,base_dir,sub_dir,filename)
            % for visaulising alignment, generate gif
                
            nframes = 10;
            
            if ~isdir([base_dir '/' sub_dir '/'])
                mkdir([base_dir '/' sub_dir '/']);
            end

            framename = [base_dir '/' sub_dir '/' filename '.gif'];

            start_end_delay = 0.5;
            normal_delay = 2.0 / nframes;

            if length(size(obj.results.im1_registered)) == 2
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
                    [imind1,cm1] = rgb2ind( repmat(uint8(im1_fract * obj.results.im2_registered + (1 - im1_fract) * obj.results.im1_registered), [1 1 3]), 256);
                else
                    [imind1,cm1] = rgb2ind( uint8(im1_fract * obj.results.im2_registered + (1 - im1_fract) * obj.results.im1_registered), 256);
                end

                if i == 1
                    imwrite(imind1, cm1, framename, 'gif', 'Loopcount', inf, 'DelayTime', fdelay);
                else
                    imwrite(imind1, cm1, framename, 'gif', 'WriteMode', 'append', 'DelayTime', fdelay);
                end

            end                

            % open gif externally of visuals toggled on
            if obj.open_gif
                unix(['gnome-open ' framename]);
            end
                
        end

    end

end