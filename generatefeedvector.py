import feedparser
import re

def getwordcounts(url):
    wc = {}
    try: 
        d = feedparser.parse(url)
    except:
        return "none", wc

    for e in d.entries:
        if 'summary' in e:
            summary = e.summary
        else:
            try:
                summary = e.description
            except:
                continue
        words = getwords(e.title + ' ' + summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1
    if hasattr(d.feed, 'title'):
        return d.feed.title, wc
    else: return "none", wc


def getwords(html):
    txt = re.compile(r'<[^>]+>').sub('', html)
    words = re.compile(r'[^A-Z^a-z]+').split(txt)
    return [word.lower() for word in words if word != '']

apcount = {}
wordcounts = {}
feedcount = 0

for feedurl in open('feedlist.txt'):
    feedcount += 1
    title, wc = getwordcounts(feedurl)
    wordcounts[title] = wc
    for word,count in wc.items():
        apcount.setdefault(word, 0)
        if count > 1:
            apcount[word] += 1

wordlist = []
for w,bc in apcount.items():
    frac = float(bc) / feedcount
    if frac > 0.1 and frac < 0.5: wordlist.append(w)

out = open('blogdata.txt', 'w')
out.write('Blog')
for word in wordlist:
    out.write('\t%s' %  word)
out.write('\n')
for blog,wc in wordcounts.items():
    out.write(blog)
    for word in wordlist:
        if word in wc: out.write('\t%d' % wc[word])
        else: out.write('\t0')
    out.write('\n')
