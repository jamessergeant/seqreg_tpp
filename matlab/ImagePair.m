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

    properties (SetAccess = private)
        im1
        im1_fullres_gray
        im1_unpad_locnorm
        im1_fullres
        im1_fullres_unpad
        im1_orig
        im1_padding
        im1_unpad
        im1_unpad_double
        im2
        im2_fullres_gray
        im2_unpad_locnorm
        im2_fullres
        im2_fullres_unpad
        im2_orig
        im2_padding
        im2_unpad
        im2_unpad_double
    end

    properties%(Access = private)
        scales
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

        function obj = ImagePair(varargin)

            obj.parseInput(varargin);

            % test if gpu available, load appropriate kernel
            obj.gpu_load();

        end
        
        function set_images(obj,im1,im2,scales)

            % set properties
            obj.im1 = im1;
            obj.im2 = im2;
            obj.scales = scales;

            % perform image preprocessing steps (scaling, patch
            % normalisation
            obj.preprocess();
            
        end
        
        function update_parameters(varargin)
            
            obj.parseInput(varargin);
            
        end
                
        function toggle_vis(obj)

            obj.visuals = ~obj.visuals;
            
        end

%     end
% 
%     methods(Access = protected)

        function parseInput(obj,in_args)
           p = inputParser;

           addParameter(p,'nps',5,@isnumeric);
           addParameter(p,'minstd',0.1,@isnumeric);
           addParameter(p,'patch_norm',false,@islogical);
           addParameter(p,'kernel_path',[fileparts(mfilename('fullpath')) '/gpu'],@isstr);
           addParameter(p,'maxdim',400,@isnumeric);
           addParameter(p,'visuals',false,@islogical);

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
                ptx_path = [obj.kernel_path '/locnorm_gpu.ptx'];
                cu_path = [obj.kernel_path '/locnorm_gpu.cu'];

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
            
            if obj.visuals
                close all
                imshowpair(obj.im1_fullres,obj.im2_fullres,'method','montage')
                pause
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

    end

end
