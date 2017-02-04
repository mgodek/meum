%
function out_set = reduce_set(factor, apriori, workSet)
    out_set=[];
    workSet_size = rows(workSet);
    for i=1:columns(apriori)
        desired_size = apriori(1, i)*workSet_size*factor;
        temp_set=workSet(workSet(:,1)==i,:);
        if desired_size > rows(temp_set)
            desired_size = rows(temp_set); % TODO moga byc problemy
        endif
        temp_set=temp_set(randperm(rows(temp_set)),:);
        out_set=[out_set; temp_set(1:desired_size,:)];
    end
endfunction