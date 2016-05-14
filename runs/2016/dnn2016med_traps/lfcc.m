function lfcc(filesPath)   
  addpath('~/ASem7/Tools/voicebox/')    
  tic;
  display(['Extracting Features from Audio Files..']);
  files = dir([filesPath, '*.wav']);


  for i = 1:length(files)
    fileName = files(i).name;      % bus_01.wav
    fileNameParts = regexp(fileName, '[a-z0-9_]+', 'match'); %'bus' 'wav'
    className = fileNameParts{1};
    display(['Reading ... FileNo : ', '[', num2str(i), '/', ...
    num2str(length(files)), '],  FileName : ', fileName]);
    [audio, sr] = audioread([filesPath fileName]);
    audio = mean(audio,2);
    features = lincepst(audio,sr,'0dD',19,40,1764);
    disp(size(features))
    save(['../../../saved/features/2016/lfcc/features/', className, '.txt'], 'features', '-ascii');
  end
  toc;
end
