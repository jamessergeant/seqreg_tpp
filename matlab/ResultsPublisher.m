classdef ResultsPublisher < handle
    
    properties
        registration_obj
        
    end
    
    methods
        
        function obj = ResultsPublisher(varargin)
            obj.registration_obj = varargin{1};
        end

%%
        function generate_fullresultshtml(obj)
        % generate the results html page showing all cases and results
            f = fopen(sprintf('%s/ind_results.html',[obj.registration_obj.save_dir '/web/' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]),'w');
            fprintf(f,obj.header_html());
            fprintf(f,'<h2>Individual Results</h2><br />');
            fprintf(f,'<table><tr><th rowspan="2" width="150">Case</th><th rowspan="2" width="150">Context</th><th colspan="2">Image Types</th><th colspan="4">Differences</th><th colspan="3">Method Results</th><th></th></tr>');
            fprintf(f,'<tr><th width="150">Image 1</th><th width="150">Image 2</th><th width="150">Translation</th><th width="150">Lighting</th><th width="150">Scale</th><th width="150">Multimodal</th><th width="150">SeqSLAM</th><th width="150">CNN</th><th width="150">SURF</th></tr>');
            string = '<tr align="center"><td>%i</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>';
            for i = 1:length(obj.registration_obj.test_cases)

                if any(obj.registration_obj.known_bad_cases == i)

                    fprintf(f,sprintf(string, ...
                        i, '<font color="red">BAD CASE</font>', ...
                        '', ...
                        '', ...
                        '', ...
                        '', ...
                        '', ...
                        '', ...
                        '', ...
                        '', ...
                        ''));

                else

                    fprintf(f,sprintf(string, ...
                        i, int2str(obj.registration_obj.test_cases(i).context), ...
                        bool2mode(obj.registration_obj.test_cases(i).Image1.mcc), ...
                        bool2mode(obj.registration_obj.test_cases(i).Image2.mcc), ...
                        bool2str(obj.registration_obj.test_cases(i).differences.translation), ...
                        bool2str(obj.registration_obj.test_cases(i).differences.lighting), ...
                        bool2str(obj.registration_obj.test_cases(i).differences.scale), ...
                        bool2str(obj.registration_obj.test_cases(i).differences.multimodal), ...
                        bool2mark(obj.registration_obj.test_cases(i).(['seqreg' num2str(obj.registration_obj.trajectory_mode)]).match), ...
                        bool2mark(obj.registration_obj.test_cases(i).cnn.match), ...
                        bool2mark(obj.registration_obj.test_cases(i).surf.match)));

                end
            end

            fprintf(f,'</table>');
            fprintf(f,'</div></body> </html>');
            fclose(f);

        end % end generate_fullresultshtml
%%
        function summ_alltesting(obj)
        % generate the category/method results table and generate the
        % summarised results page

            obj.registration_obj.type_table = unique(combntns([0 1 0 1 0 1 0 1 0 1 0 1],6),'rows');

            obj.registration_obj.summ_testing('cnn');
            Total = obj.registration_obj.test_totals;
            ind = Total ~= 0;
            Total = Total(ind);
            obj.registration_obj.type_table = obj.registration_obj.type_table(ind,:);
            CNN = obj.registration_obj.results.([obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]);
            CNN = CNN(ind);
            obj.registration_obj.results.([obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]) = CNN;

            obj.registration_obj.summ_testing('surf');
            assert(all(Total(:) == obj.registration_obj.test_totals(:)));
            SURF = obj.registration_obj.results.([obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]);

            obj.registration_obj.summ_testing('seqreg');
            assert(all(Total(:) == obj.registration_obj.test_totals(:)));
            SeqSLAM = obj.registration_obj.results.([obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]);

            Image1 = obj.registration_obj.type_table(:,1);
            Image2 = obj.registration_obj.type_table(:,2);
            Translation = obj.registration_obj.type_table(:,3);
            Lighting = obj.registration_obj.type_table(:,4);
            Scale = obj.registration_obj.type_table(:,5);
            Multimodal = obj.registration_obj.type_table(:,6);

            obj.registration_obj.results.table = table(Image1,Image2,Translation,Lighting,Scale,Multimodal,SeqSLAM,CNN,SURF,Total);
            obj.registration_obj.results.table

            obj.registration_obj.save_prog();

            obj.generate_home();

        end

        function string = header_html(obj)

            script = sprintf('<script> function process() { var url="file://%s/pages/" + document.getElementById("url").value + ".html"; location.href=url; return false; } </script>',[obj.registration_obj.save_dir '/web-' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]);
            string = sprintf('<!DOCTYPE html> <html lang="en"> <head> <meta charset="utf-8"> <title>QUT / NASA Image Registration</title> <link rel="stylesheet" href="%s/css/style.css"> <script src="script.js"></script> %s</head> <body><div align="center"><br /><table><tr align="center"><td width="200"><img src="/home/james/Dropbox/NASA/SeqSLAM/acrv.jpg" height="50" /></td><td width="200"><img src="/home/james/Dropbox/NASA/SeqSLAM/qut.png" height="50" /></td><td width="200"><img src="/home/james/Dropbox/NASA/SeqSLAM/nasa-jpl.gif" height="50" /></td></tr></table><br /><h1><a href="%s/index.html">QUT / NASA Image Registration</a><br /></h1><form onSubmit="return process();">Go to test case: <input  type="text" name="url" id="url"><input type="submit" value="Go"></form><br />',[obj.registration_obj.save_dir '/web-' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)],script,[obj.registration_obj.save_dir '/web-' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]);

        end
%%
        function generate_home(obj)

            f = fopen(sprintf('%s/index.html',[obj.registration_obj.save_dir '/web/' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)]),'w');
            fprintf(f,obj.header_html());
            fprintf(f,'<h2>Current Results</h2><br />');
            fprintf(f,'<table><tr><th colspan="2">Image Types</th><th colspan="4">Differences</th><th colspan="3">Method Results</th><th></th></tr>');
            fprintf(f,'<tr><th width="150">Image 1</th><th width="150">Image 2</th><th width="150">Translation</th><th width="150">Lighting</th><th width="150">Scale</th><th width="150">Multimodal</th><th width="150">SeqSLAM</th><th width="150">CNN</th><th width="150">SURF</th><th width="150">Total</th></tr>');
            string = '<tr align="center"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%i (%0.1f%s)</td><td>%i (%0.1f%s)</td><td>%i (%0.1f%s)</td><td>%i</td></tr>';
            for i = 1:size(obj.registration_obj.results.table,1)

                fprintf(f,sprintf(string, ...
                    bool2mode(obj.registration_obj.results.table.Image1(i)), ...
                    bool2mode(obj.registration_obj.results.table.Image2(i)), ...
                    bool2str(obj.registration_obj.results.table.Translation(i)), ...
                    bool2str(obj.registration_obj.results.table.Lighting(i)), ...
                    bool2str(obj.registration_obj.results.table.Scale(i)), ...
                    bool2str(obj.registration_obj.results.table.Multimodal(i)), ...
                    obj.registration_obj.results.table.SeqSLAM(i), ...
                    obj.registration_obj.results.table.SeqSLAM(i) * 100 / obj.registration_obj.results.table.Total(i), '%%',...
                    obj.registration_obj.results.table.CNN(i), ...
                    obj.registration_obj.results.table.CNN(i) * 100 / obj.registration_obj.results.table.Total(i), '%%', ...
                    obj.registration_obj.results.table.SURF(i), ...
                    obj.registration_obj.results.table.SURF(i) * 100 / obj.registration_obj.results.table.Total(i), '%%', ...
                    obj.registration_obj.results.table.Total(i)));
            end

            string = '<tr align="center"><th></th><th></th><th></th><th></th><td></th><th></th><th>%i (%0.1f%s)</th><th>%i (%0.1f%s)</th><th>%i (%0.1f%s)</th><th>%i</th></tr>';

            fprintf(f,sprintf('<br />%s',sprintf(string, ...
                sum(obj.registration_obj.results.table.SeqSLAM), ...
                sum(obj.registration_obj.results.table.SeqSLAM) * 100 / sum(obj.registration_obj.results.table.Total), '%%', ...
                sum(obj.registration_obj.results.table.CNN), ...
                sum(obj.registration_obj.results.table.CNN) * 100 / sum(obj.registration_obj.results.table.Total), '%%', ...
                sum(obj.registration_obj.results.table.SURF), ...
                sum(obj.registration_obj.results.table.SURF) * 100 / sum(obj.registration_obj.results.table.Total), '%%', ...
                sum(obj.registration_obj.results.table.Total))));
            fprintf(f,'</table>');
            fprintf(f,'</div></body> </html>');
            fclose(f);

        end
%%
        function generate_html(obj)

            root = [obj.registration_obj.save_dir '/web/' obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode)];

            tc = obj.registration_obj.test_cases(obj.registration_obj.curr_case);
            
            f = fopen(sprintf('%s/pages/%i.html',root,obj.registration_obj.curr_case),'w');
            fprintf(f,obj.header_html());
            fprintf(f,sprintf('<h2>Context %i - Test Case %i</h2><br />', tc.context, obj.registration_obj.curr_case));
            if any(obj.registration_obj.known_bad_cases == obj.registration_obj.curr_case)
                fprintf(f,'<h2><font color="red">BAD CASE</font></h2><br />');
            end
            fprintf(f,sprintf('<h4><a href="%i.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;<a href="%i.html">Next</a></h4><br />',max(obj.registration_obj.curr_case-1,1),min(obj.registration_obj.curr_case+1,length(obj.registration_obj.test_cases))));
            fprintf(f,'<table><tr><th></th><th width="150">Image 1</th><th width="150">Image 2</th><th width="150">Differences</th></tr>');
            tx1 = tc.Image1.Translation_X;
            tx2 = tc.Image2.Translation_X;
            ty1 = tc.Image1.Translation_Y;
            ty2 = tc.Image2.Translation_Y;
            fprintf(f,sprintf('<tr><th>Translation X</th><td align="center">%0.1f</td><td align="center">%0.1f</td><td rowspan="2" align="center">%s</td></tr>', tx1, tx2, bool2str(tc.differences.translation)));
            fprintf(f,sprintf('<tr><th>Translation Y</th><td align="center">%0.1f</td><td align="center">%0.1f</td></tr>', ty1, ty2));
            lighting = {'Top','Right','Left'};
            fprintf(f,sprintf('<tr><th>Illumination</th><td align="center">%s</td><td align="center">%s</td><td align="center">%s</td></tr>', lighting{tc.Image1.Illumination}, lighting{tc.Image2.Illumination}, bool2str(tc.differences.lighting)));
            fprintf(f,sprintf('<tr><th>Focal Extension (Scale)</th><td align="center">%0.1fmm</td><td align="center">%0.1fmm</td><td align="center">%s</td></tr>', tc.Image1.Extension, tc.Image2.Extension, bool2str(tc.differences.scale)));
            image_modes = {'Watson','PIXL'};
            fprintf(f,sprintf('<tr><th>Mode</th><td align="center">%s</td><td align="center">%s</td><td align="center">%s</td></tr>', image_modes{tc.Image1.mcc+1}, image_modes{tc.Image2.mcc+1}, bool2str(tc.differences.multimodal)));
            fprintf(f,'</table><br /><br /><br />');


            fprintf(f,'<table><tr><th width="400">SeqSLAM</th><th width="400">CNN</th><th width="500">SURF</th></tr><tr>');
            modes = {'seqreg','cnn','surf'};
            for i = 1:3
                image = sprintf('%s%s_%i.gif',[obj.registration_obj.gifdir obj.registration_obj.method num2str(obj.registration_obj.trajectory_mode) '/'],modes{i},obj.registration_obj.curr_case);
                if exist(image, 'file') == 2
                    fprintf(f,sprintf('<td align="center"><a href="%s"><img src="%s" width="500" border="5" style="border-color: %s"/></a></td>',image,image,match2colour(obj.registration_obj.test_cases(obj.registration_obj.curr_case).([modes{i} num2str(obj.registration_obj.trajectory_mode)]).match)));
                else
                    fprintf(f,'<td align="center">No GIF</td>');
                end

            end
            
            fprintf(f,'</tr><tr>');
            
            for i = 1:3
                if strcmp(modes{i},'seqreg')
                    add_mode = num2str(obj.registration_obj.trajectory_mode);
                else
                    add_mode = '';
                end
                if isfield(tc.([modes{i} add_mode]),'tform_status') & tc.([modes{i} add_mode]).tform_status == 0

                    fprintf(f,sprintf('<td align="center"><ul><li>Scale: %0.3f</li><li>Rotation: %0.3f</li></td>',tc.([modes{i} add_mode]).scaleRecovered,tc.([modes{i} add_mode]).thetaRecovered));
                else
                    fprintf(f,'<td align="center">No scale or rotation</td>');
                end

            end
            fprintf(f,'</tr></table>');
            fprintf(f,'</div></body> </html>');
            fclose(f);
        end
%%
        function generate_allhtml(obj,varargin)
            if nargin > 1
                if ~isscalar(varargin{1})
                    items = varargin{1};
                else
                    if varargin{1} < varargin{2}
                        items = varargin{1}:varargin{2};
                    else
                        items = varargin{2}:-1:varargin{2};
                    end
                end
            else
                items = 1:length(obj.registration_obj.test_cases);
            end

            for i=items

                obj.registration_obj.curr_case = i;
                obj.generate_html();

            end
        end
%%
        function update_home(obj)

            obj.summ_alltesting();
            obj.generate_home();

        end
        
    end
end

function string = bool2str(b)

string = regexprep(sprintf('%i',boolean(b)),{'1','0'},{'True','False'});

end

function string = bool2mode(b)

string = regexprep(sprintf('%i',boolean(b)),{'1','0'},{'PIXL','Watson'});

end


function string = match2colour(b)

string = regexprep(sprintf('%i',boolean(b)),{'1','0'},{'green','red'});

end

% convert boolean to green tick or red cross
function string = bool2mark(b)

if boolean(b)
    string = '<font color="green">&#10004;</font>';
else
    string = '<font color="red">&#10008;</font>';
end

end