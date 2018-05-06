./tree -train wiki.txt -output wikiout
./cal -train wiki.txt -read-vocab wikiout2 -save-vocab wikiout2context
./processtxt -train wiki.txt -read-vocab wikiout2context -output tagwiki.txt
./tree -train tagwiki.txt -output tagwikiout
./caltop < tagwikiout2 > synresult
