%% Shifted linear interpolation of the Gaussian function
% Reference:
% Blu T, Thévenaz P, Unser M (2004): Linear Interpolation Revitalized.
% IEEE Trans. Image Proc. 13(5), 710-9

clear all; close all; clc;

dx = 0.25;

% vector of samples
xk = -5:dx:5;
fk = exp(-xk.^2/2);

figure; plot(xk,fk,'o-'); xlim([-5 5]);
print -dpng gauss01.png; close;


%% === Prefilter ===
function c = prefilter(fk,shift)
    pole = shift/(shift-1);
    zero = 1/(1-shift);

    c = zeros(size(fk));
    c(1) = fk(1);
    for i = 2:length(c)
        c(i) = pole*c(i-1) + zero*fk(i);
    end
end


%% === Shifted linear interpolation for arbitrary x ===
function y = slintp(x,xks,c)
    dx = xks(2)-xks(1);
    N = length(c);

    % left boundary
    if x <= xks(1)
        y = c(1);
        return;
    end

    nk = (x - xks(1))/dx;
    k = floor(nk);

    % Clamp k so we never access c(k+2) out of range
    if k >= N-1
        y = c(end);
        return;
    end

    deltax = dx*(nk - k);

    % Python: c[k]*(dx-deltax) + c[k+1]*deltax
    % MATLAB: c(k+1)*(dx-deltax) + c(k+2)*deltax
    y = (c(k+1)*(dx-deltax) + c(k+2)*deltax) / dx;
end



%% ==== shifted linear interpolation ====

shift = 0.21;
xks = xk + shift*dx;

xi  = -5:0.01:5;
nxi = length(xi);
fi  = zeros(size(xi));

c = prefilter(fk,shift);

for i = 1:nxi
    fi(i) = slintp(xi(i),xks,c);
end

dfis = exp(-xi.^2/2) - fi;

figure; plot(xi,dfis); xlim([-5 5]);
print -dpng gauss02.png; close;

L2s = std(dfis);
fprintf("std: %.6f\n",L2s);


%% ==== not shifted linear interpolation ====

shift = 0.0;
xks = xk + shift*dx;
c  = prefilter(fk,shift);

for i = 1:nxi
    fi(i) = slintp(xi(i),xks,c);
end

dfil = exp(-xi.^2/2) - fi;

figure; plot(xi,dfil); xlim([-5 5]);
print -dpng gauss03.png; close;

L2l = std(dfil);
fprintf("std: %.6f\n",L2l);

dBgain = 20*log10(L2l/L2s);
fprintf("gain: %.3f\n",dBgain);


%% ==== interpolation gain as a function of shift ====

function g = gain(shift)
    dx = 0.25;
    xk = -5:dx:5;
    fk = exp(-xk.^2/2);

    xks = xk + shift*dx;

    xi = -5:0.01:5;
    nxi = length(xi);

    c = prefilter(fk,shift);
    fi = zeros(size(xi));

    for j = 1:nxi
        fi(j) = slintp(xi(j),xks,c);
    end

    df = exp(-xi.^2/2) - fi;
    L2s = std(df);
    L2l = 0.002072;

    g = 20*log10(L2l/L2s);
end


s = 0:0.01:0.5;
gs = zeros(size(s));

for i = 1:length(s)
    gs(i) = gain(s(i));
end

figure; plot(s,gs); xlim([0 0.5]);
print -dpng gauss04.png; close;


%% ===== h(n) prefilter coefficients =====

tau = 0.21;
kvec = 0:100;

hn = ((-1).^kvec)/(1-tau) .* (tau/(1-tau)).^kvec;

figure; plot(kvec(1:21), hn(1:21));
xlim([0 20]);
print -dpng gauss05.png; close;


%% ===== hat function =====
function y = hat(x)
    if abs(x) > 1
        y = 0;
    else
        y = 1 - abs(x);
    end
end


%% ==== synthesis function ====
function f = fint(x,hn,tau)
    f = 0;
    n = length(hn);

    for i = 1:n
        f = f + hn(i) * hat(x - i + 1 - tau);
    end
end


xf = 0:0.05:10;
f  = zeros(size(xf));

for i = 1:length(xf)
    f(i) = fint(xf(i),hn,tau);
end

figure; plot(xf,f); hold on;
x0 = 0:10;
z0 = zeros(size(x0)); z0(1)=1;
plot(x0,z0,'ro');
xlim([-0.08,10.08]);
print -dpng gauss06.png;
close;

