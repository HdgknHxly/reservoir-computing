function delayCoord = delayEmbedding(data, edim, delay)
% 時系列データから時間遅延座標系を構成する関数
%
% delayCoord = delayEmbedding(data, edim, delay)
%
% Parameters
% ----------
% data: array [length(data), nChannels]
%   時系列データ
% edim: int
%   埋め込み次元
% delay: int
%   遅延長
%
% Returns
% -------
% delayCoord: array [length(data) - (edim-1)*delay, edim, nChannels]
%   時間遅延座標系
%
    lenDelayCoord = length(data) - (edim-1)*delay;  % 遅延座標系のデータ長
    nChannels = width(data);  % 入力データのチャンネル数
    delayCoord = zeros(lenDelayCoord, edim, nChannels);
    for j = 1:nChannels
        for i = 1:edim
            delayCoord(:, i, j) = data(1+(i-1)*delay:end-(edim-i)*delay, j);
        end
    end
end
