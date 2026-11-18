(function () {
  var S = [["1.01", "microgpt"], ["1.02", "PyTorch"], ["1.03", "Batch"], ["1.04", "GPU"], ["1.05", "Tokenizer"], ["1.06", "优化器"], ["1.07", "学习率调度"], ["1.08", "数值精度"], ["1.09", "工程加速"], ["1.10", "训练细节"], ["1.11", "回顾"]];
  var DONE = ["1.01", "1.02", "1.03", "1.04", "1.05", "1.06", "1.07", "1.08", "1.09"];
  var a = document.getElementById('side'), cur = a.dataset.cur, sub = a.querySelector('.subs');
  var subs = sub ? sub.innerHTML : '';
  var h = '<div class="ch"><a href="./#chapters">第一章 · 训练</a></div><ol>';
  S.forEach(function (s) {
    var n = s[0], t = '<span class="sn">' + n + '</span>' + s[1];
    if (n === cur) h += '<li><a class="cur" href="' + n + '.html">' + t + '</a><ol>' + subs + '</ol></li>';
    else if (DONE.indexOf(n) >= 0) h += '<li><a href="' + n + '.html">' + t + '</a></li>';
    else h += '<li><span class="off">' + t + '</span></li>';
  });
  a.innerHTML = h + '</ol>';
})();
