function idx = nmsselect(idx, score, heilen, widlen)
    tmpidx = idx;
    len = length(tmpidx);
    
    % 20160510 only keep score for face locations, other positions set to 0
    localscore = zeros(size(score));
    localscore(tmpidx) = score(tmpidx);
    for i = 1:len
        centeridx = tmpidx(i);
        [centerRow, centerCol] = ind2sub([heilen widlen], centeridx);
        if centerRow > 1 && centerRow < heilen
            up = centeridx - 1;
            bot = centeridx + 1;
        elseif centerRow == 1
            up = [];
            bot = centeridx + 1;
        else
            up = centeridx - 1;
            bot = [];
        end
        
        if centerCol > 1 && centerCol < widlen
            left = centeridx - heilen;
            right = centeridx + heilen;
        elseif centerCol == 1
            left = [];
            right = centeridx + heilen;
        else
            left = centeridx - heilen;
            right = [];
        end
        
        neighboridx = [up bot left right];
        % if center is not the maximum in the neigborhood, delete it
        if sum(localscore(centeridx) >= localscore(neighboridx)) ~= length(neighboridx)
           idx = setdiff(idx, centeridx);
        end
        
    end

end