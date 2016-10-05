classdef ImagePair < handle
    %IMAGEPAIR The pair of images and related methods for use with the
    % ImageRegistration class.
    %   Input Arguments:
    %       im1 & im2:  RGB or greyscale images to be registered.
    %       scales:     A 2-element vector containin the relative scales of
    %                   the two images.
    %
    %   Public Methods:
    %       None.

    properties
        im1
        im1_fullres_gray
        im1_unpad_locnorm
        im2
        im2_fullres_gray
        im1_fullres
        im2_unpad_locnorm
        im1_fullres_unpad
        im2_fullres_unpad
        im1_orig
        im2_orig
        im2_fullres
    end

    properties(Access = private)
        scales
        im1_padding
        im1_unpad
        im1_unpad_double
        im2_padding
        im2_unpad
        im2_unpad_double
        gpu_processing
        GPU
        block_size
        kernel_path
        gpu_locnorm
        gpu_out
        maxdim
        xres
        yres
        patch_norm
        nps
        minstd
        visuals
    end

    methods

        function obj = ImagePair(im1,im2,scales,varargin)

            obj.parseInput(varargin);

            % set properties
            obj.im1 = im1;
            obj.im2 = im2;
            obj.scales = scales;

            % test if gpu available, load appropriate kernel
            obj.gpu_load();

            % perform image preprocessing steps (scaling, patch
            % normalisation
            obj.preprocess();

        end

        % modify set and get properties for the various images
%         function set.im1(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im1(obj)
            im_out = obj.im1;
        end

%         function set.im1_fullres_gray(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im1_fullres_gray(obj)
            im_out = obj.im1_fullres_gray;
        end

%         function set.im1_unpad_locnorm(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im1_unpad_locnorm(obj)
            im_out = obj.im1_unpad_locnorm;
        end

%         function set.im2(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im2(obj)
            im_out = obj.im2;
        end

%         function set.im2_fullres_gray(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im2_fullres_gray(obj)
            im_out = obj.im2_fullres_gray;
        end

%         function set.im2_unpad_locnorm(~,~)
%             warning('You cannot set this property');
%         end

        function im_out = get.im2_unpad_locnorm(obj)
            im_out = obj.im2_unpad_locnorm;
        end

        function im_out = get.im1_fullres_unpad(obj)
            im_out = obj.im1_fullres_unpad;
        end

        function im_out = get.im2_fullres_unpad(obj)
            im_out = obj.im2_fullres_unpad;
        end

        function im_out = get.im1_fullres(obj)
            im_out = obj.im1_fullres;
        end

        function im_out = get.im2_fullres(obj)
            im_out = obj.im2_fullres;
        end

        function im_out = get.im1_orig(obj)
            im_out = obj.im1_orig;
        end

        function im_out = get.im2_orig(obj)
            im_out = obj.im2_orig;
        end
                
        function toggle_vis(obj)

            obj.visuals = ~obj.visuals;
            
        end

    end

    methods(Access = protected)

        function parseInput(obj,in_args)
           p = inputParser;

           addOptional(p,'nps',5,@isnumeric);
           addOptional(p,'minstd',0.1,@isnumeric);
           addOptional(p,'patch_norm',false,@isbool);
           addOptional(p,'kernel_path','/home/james/co/SeqSLAM_GPU',@isstr);
           addOptional(p,'maxdim',400,@isnumeric);
           addOptional(p,'visuals',false,@isbool);

           parse(p,in_args{:});

           fields = fieldnames(p.Results);

           for i = 1:numel(fields)
               obj.(fields{i}) = p.Results.(fields{i});
           end

        end

        function gpu_load(obj)

            obj.gpu_processing = false;

            for i=1:gpuDeviceCount
                if parallel.gpu.GPUDevice.isAvailable(i)
                    obj.GPU = parallel.gpu.GPUDevice.getDevice(i);
                    obj.gpu_processing = true;
                    disp([class(obj) ': GPU found!']);
                    break;
                end
            end

            if obj.gpu_processing
                ptx_path = [obj.kernel_path '/SeqSLAM_kernel.ptx'];
                cu_path = [obj.kernel_path '/SeqSLAM_kernel.cu'];

                obj.block_size = floor(sqrt(obj.GPU.MaxThreadsPerBlock));

                obj.gpu_locnorm = parallel.gpu.CUDAKernel(ptx_path,cu_path,'_Z10local_normPKfPfiiif');
                obj.gpu_locnorm.ThreadBlockSize = [obj.block_size obj.block_size];

                wait(obj.GPU);

            end

        end

        function preprocess(obj)
        % load and preprocess images for current case, including patch
        % normalisation if currently specified

            disp('Loading images...');

            % multimodal cases, greyscale images converted to 3 channel
            % greyscale,
            if length(size(obj.im1)) ~= length(size(obj.im2))
                if length(size(obj.im1)) == 2
                    obj.im1 = repmat(obj.im1,[1 1 3]);
                    obj.im1 = imresize(obj.im1,[size(obj.im2,1) nan]);
                end
                if length(size(obj.im2)) == 2
                    obj.im2 = repmat(obj.im2,[1 1 3]);
                    obj.im2 = imresize(obj.im2,[size(obj.im1,1) nan]);
                end
            end

            % convert to uint8
            obj.im1 = im2uint8(obj.im1);
            obj.im2 = im2uint8(obj.im2);

            % pre-scale images
            obj.scales = min(obj.scales) ./ obj.scales;

            obj.scales = obj.scales;

            obj.im1_fullres_unpad = obj.im1;
            obj.im2_fullres_unpad = obj.im2;

            for i = 1:length(obj.scales)
                
                obj.(['im' num2str(i)]) = imresize(obj.(['im' num2str(i)]),obj.scales(i));
                obj.(['im' num2str(i) '_unpad']) = obj.(['im' num2str(i)]);
                
                s1 = size(obj.(['im' num2str(i)]));
                s2 = size(obj.(['im' num2str(3-i)]));

                % add black padding to match other image size
                obj.(['im' num2str(i) '_padding']) = [max(0,round((s2(1) - s1(1))/2)), max(0,round((s2(2) - s1(2))/2))];
                obj.(['im' num2str(i)]) = padarray(obj.(['im' num2str(i)]),obj.(['im' num2str(i) '_padding']));

                obj.(['im' num2str(i)]) = obj.(['im' num2str(i)])(1:s2(1),1:s2(2),:);

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
            obj.im1_unpad = imresize(obj.im1_unpad, [obj.scales(1)*obj.yres obj.scales(1)*obj.xres]);
            obj.im1_padding = ceil(obj.im1_padding * padding_scale);
            obj.im2_orig = imresize(obj.im2_fullres, [obj.yres obj.xres]);
            obj.im2_fullres_unpad = obj.im2_unpad;
            obj.im2_unpad = imresize(obj.im2_unpad, [obj.scales(2)*obj.yres obj.scales(2)*obj.xres]);
            obj.im2_padding = ceil(obj.im2_padding * padding_scale);

            % convert im1 to double, greyscale
            if length(size(obj.im1_orig)) == 2
                obj.im1 = double(obj.im1_orig);
                obj.im1_unpad_double = double(obj.im1_unpad);
            else
                obj.im1 = double(rgb2gray(obj.im1_orig));
                obj.im1_unpad_double = double(rgb2gray(obj.im1_unpad));
            end

            % convert im2 to double, greyscale
            if length(size(obj.im2_orig)) == 2
                obj.im2 = double(obj.im2_orig);
                obj.im2_unpad_double = double(obj.im2_unpad);
            else
                obj.im2 = double(rgb2gray(obj.im2_orig));
                obj.im2_unpad_double = double(rgb2gray(obj.im2_unpad));
            end

            % finished loading images
            fprintf('...done\n');

            disp('Patch normalising...');

            if obj.gpu_processing
                obj.im1 = obj.process_locnorm(obj.gpu_locnorm, obj.im1, obj.block_size,obj.nps,obj.minstd);
                obj.im1_unpad_locnorm = obj.process_locnorm(obj.gpu_locnorm, obj.im1_unpad_double, obj.block_size,obj.nps,obj.minstd);
                obj.im2 = obj.process_locnorm(obj.gpu_locnorm, obj.im2, obj.block_size,obj.nps,obj.minstd);
                obj.im2_unpad_locnorm = obj.process_locnorm(obj.gpu_locnorm, obj.im2_unpad_double, obj.block_size,obj.nps,obj.minstd);
            else
                obj.im1 = locnorm(obj.im1, obj.nps, obj.minstd);
                obj.im1_unpad_locnorm = locnorm(obj.im1_unpad_double,obj.nps,obj.minstd);
                obj.im2 = locnorm(obj.im2, obj.nps, obj.minstd);
                obj.im2_unpad_locnorm = locnorm(obj.im2_unpad_double,obj.nps,obj.minstd);
            end

            fprintf('...done\n');

            if obj.visuals
                close all
                imshowpair(obj.im1_fullres,obj.im2_fullres,'method','montage')
                disp('Hit any key to continue...')
                pause
                close all
            end

        end % end im_load

        function im_out = process_locnorm(obj,func,image,block_size,nps,minstd)
            gpu_im_out = zeros(size(image),'single','gpuArray');
            gpu_im = gpuArray(im2single(image));
            [height, width] = size(image);
            func.GridSize = [ceil(height / block_size),ceil(width / block_size),1];
            gpu_im_out = feval(func,gpu_im,gpu_im_out,width,height,nps,minstd);
            im_out = gather(gpu_im_out);
            wait(obj.GPU)
        end


%         function save_im(obj)
%         % generate gif using the estimated transform
%
%             % test if the current case has successfully estimated a
%             % transform
%             if ~obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).match
%                 return
%             end
%
%             % depending on method, use padded or unpadded image
%             if strcmp(obj.method,'cnn')
%                 im1f = obj.im1_fullres_unpad;
%                 im2f = obj.im2_fullres_unpad;
%             else
%                 im1f = obj.im1_fullres;
%                 im2f = obj.im2_fullres;
%             end
%
%             % create number of frames, desired dimension
%             nframes = 10;
%             desdim = 800;
%
%             % different methods use different fixed and moving images,
%             % should update this to make consistent
%             if ~strcmp(obj.method,'cnn')
%                 [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(im2f,2)], [1 size(im2f,1)]);
%
%                 % Find the minimum and maximum output limits
%                 xMin = min([1; xLimitsOut(:)]);
%                 xMax = max([size(im1f,2); xLimitsOut(:)]);
%                 yMin = min([1; yLimitsOut(:)]);
%                 yMax = max([size(im1f,1); yLimitsOut(:)]);
%
%             else
%                 [xLimitsOut,yLimitsOut] = outputLimits(obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, [1 size(im1f,2)], [1 size(im1f,1)]);
%
%                 % Find the minimum and maximum output limits
%                 xMin = min([1; xLimitsOut(:)]);
%                 xMax = max([size(im2f,2); xLimitsOut(:)]);
%                 yMin = min([1; yLimitsOut(:)]);
%                 yMax = max([size(im2f,1); yLimitsOut(:)]);
%
%             end
%
%             % Width, height and limits of outView
%             width  = round(xMax - xMin);
%             height = round(yMax - yMin);
%             xLimits = [xMin xMax];
%             yLimits = [yMin yMax];
%             outView = imref2d([height width], xLimits, yLimits);
%
%             % apply identity transform to im2 to match size
%             obj.im2_registered = imwarp(im2f, affine2d(eye(3)), 'OutputView', outView);
%
%             % apply estimated transform to im1
%             registered = imwarp(im1f, obj.test_cases(obj.curr_case).([obj.method num2str(obj.trajectory_mode)]).tform, 'OutputView', outView);
%
%             % remove any excess padding from both images
%             mask = obj.im2_registered == 0 & registered == 0;
%             mask = prod(mask,3);
%             mask_test = double(repmat(all(mask'),[size(mask,2),1]))' | double(repmat(all(mask),[size(mask,1),1]));
%             obj.im2_registered = obj.im2_registered(~all(mask_test'),~all(mask_test),:);
%             registered = registered(~all(mask_test'),~all(mask_test),:);
%
%             % resize to max dimension of desdim
%             max_dim = max(size(registered));
%             r_dim = [NaN NaN];
%             r_dim(size(registered) == max_dim) = desdim;
%             obj.im1_registered = imresize(registered,r_dim);
%             obj.im2_registered = imresize(obj.im2_registered,r_dim);
%
%             % Generate gif
%             outstring = sprintf('%s_%i',obj.method,obj.curr_case);
%
%             if ~isdir([obj.gifdir obj.method num2str(obj.trajectory_mode) '/' '/'])
%                 mkdir([obj.gifdir obj.method num2str(obj.trajectory_mode) '/' '/']);
%             end
%
%             framename = sprintf('%s%s.gif',[obj.gifdir obj.method num2str(obj.trajectory_mode) '/'],outstring)
%
%             start_end_delay = 0.5;
%             normal_delay = 2.0 / nframes;
%
%             if length(size(obj.im1_registered)) == 2
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
%
%             % open gif externally of visuals toggled on
%             if obj.gif_visuals
%                 unix(['gnome-open ' framename]);
%             end
%
%             % regenerate the results page for this case
%             obj.results_publisher = ResultsPublisher(obj);
%             obj.results_publisher.generate_html();
%
%         end


    end

end
