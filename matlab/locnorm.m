function [newim] = locnorm(im1, nps, min_std)

imsz = size(im1);

newim = zeros(size(im1));

for x = 1:imsz(2)
    
    for y = 1:imsz(1)
        
        xr = [max(x - nps, 1) min(x + nps, imsz(2))];
        yr = [max(y - nps, 1) min(y + nps, imsz(1))];
        
        sp1 = im1(yr(1):yr(2), xr(1):xr(2));                
        mn1 = mean(sp1(:));
        std1 = std(sp1(:));
 
        newim(y, x) = (im1(y, x) - mn1) / std1;
        
        if std1 < min_std
            newim(y, x) = 0;
        end
        
    end
    
end