function [Qzg, xi] = getQz(xap, W, b, Qx, Kz, Kw, Ts)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
n = length(xap);

dgdx = dg_dx(xap, W, b)';

ne = size(Kz,1);

if ne == n
    xi = [];
    Qzg = ((dgdx')\Qx)/dgdx;
else
    %% Algorithm to find Q according to phi
    [V,D,W] = eig(Kz);
    eigs = diag(D);

    % Get gilbert controllability
    [sys_c, ~] = canon(ss(Kz,Kw,eye(ne),zeros(ne,size(Kw,2)),Ts),'modal');

    Btilde = V\sys_c.B;

    if size(Kw,2) > 1
        ctbr_ranking = max(abs(Btilde)')';
    else
        ctbr_ranking = abs(Btilde);
    end

    ctbr_ranking = round(ctbr_ranking,6);

    i = 1;
    while i < length(ctbr_ranking)
        if ctbr_ranking(i) == ctbr_ranking(i+1)
            ctbr_ranking(i) = 2 * ctbr_ranking(i);
            ctbr_ranking(i+1) = 2 * ctbr_ranking(i+1);
            i = i + 2;
        else
            i = i + 1;
        end
    end

    [~, indicies] = sort(ctbr_ranking);
    i_min_ctrb_gilbert = indicies(ne-n+1:end);

    % Lückel controllabilty
    sys = (ss(Kz,Kw,eye(ne),zeros(ne,size(Kw,2)),5));
    [~,Dc,Wc] = eig(sys.A);
    xi = [];
    for i = 1:ne
        xi = [xi, conj(Wc(:,i))'*sys.B*sys.B'*Wc(:,i)/(conj(Wc(:,i))'*Wc(:,i))];%*exp(1-abs(real(Dc(i,i))))];
    end

    xi = round(abs(xi)*1e10,4);

    i = 1;
    while i < length(xi)
        if xi(i) == xi(i+1)
            xi(i) = 2 * xi(i);
            xi(i+1) = 2 * xi(i+1);
            i = i + 2;
        else
            i = i + 1;
        end
    end

    [~, indicies] = sort(abs(xi));
    i_min_ctrb_lueckel = indicies(ne-n+1:end);

    % Get Qz
    W1 = W(:,i_min_ctrb_lueckel)';
    Qzg = real(W1'*inv(dgdx'*W1')*Qx*inv(W1*dgdx)*W1);

    selected_eigenfunc = [i_min_ctrb_gilbert';i_min_ctrb_lueckel];
    xi = [diag(Dc), ctbr_ranking, xi'];
end
end

