function [tlab, tvec] = readmnist(datafn, labelfn)
% function reads mnist data and labels 

[fid, msg] = fopen(datafn, 'rb');

if fid==-1
   error('Error opening data file "%s" %s', datafn, msg);
end;
fseek(fid, 0, 'eof');
cnt = (ftell(fid) - 16)/784;

fseek(fid, 16, 'bof');
tvec=zeros(cnt, 784);

for i=1:cnt
   im = fread(fid, 784, 'uchar');
   tvec(i,:) = (im(:)/255.0)';
end;
fclose(fid);
cnt;

fid = fopen(labelfn, 'rb');
if fid==-1
   error('Error opening label file');
end;
fseek(fid, 8, 'bof');
[tlab nel] = fread(fid, cnt, 'uchar');
if nel ~= cnt 
   disp('Not all elements read.');
end;
fclose(fid);
nel;
