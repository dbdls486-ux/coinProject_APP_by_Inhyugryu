// 자료 출처 접속 시험 — 어떤 과거 뉴스·상장 자료를 받을 수 있는지 확인한다.
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";
const urls = [
  ["upbit candles", "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=2"],
  ["upbit candles old", "https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=2&to=2018-01-05%2000:00:00"],
  ["upbit markets", "https://api.upbit.com/v1/market/all?is_details=true"],
  ["upbit notices", "https://api-manager.upbit.com/api/v1/announcements?os=web&page=1&per_page=5&category=trade"],
  ["bithumb notices", "https://api.bithumb.com/v1/notices?page=1&count=5"],
  ["bithumb feed", "https://feed.bithumb.com/notice"],
  ["coindesk news", "https://data-api.coindesk.com/news/v1/article/list?lang=EN&limit=3"],
  ["coindesk news old", "https://data-api.coindesk.com/news/v1/article/list?lang=EN&limit=3&to_ts=1600000000"],
  ["coindesk sources", "https://data-api.coindesk.com/news/v1/source/list?lang=EN"],
  ["cryptocompare news", "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&lTs=1600000000"],
  ["blockmedia wp", "https://www.blockmedia.co.kr/wp-json/wp/v2/posts?per_page=3&_fields=id,date_gmt,title"],
  ["blockmedia wp old", "https://www.blockmedia.co.kr/wp-json/wp/v2/posts?per_page=3&_fields=id,date_gmt,title&before=2021-01-01T00:00:00"],
  ["cryptoslate wp", "https://cryptoslate.com/wp-json/wp/v2/posts?per_page=3&_fields=id,date_gmt,title"],
  ["coinness", "https://api.coinness.com/feed/v1/news?limit=3"],
  ["gdelt", "https://api.gdeltproject.org/api/v2/doc/doc?query=upbit%20listing&mode=artlist&maxrecords=3&format=json&startdatetime=20210101000000&enddatetime=20210301000000"],
  ["binance announce", "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=48&pageNo=1&pageSize=3"],
  ["coinbase candles", "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400&start=2018-01-01T00:00:00Z&end=2018-01-05T00:00:00Z"],
  ["coinbase products", "https://api.exchange.coinbase.com/products"],
  ["llama raises", "https://api.llama.fi/raises"],
  ["llama hacks", "https://api.llama.fi/hacks"],
  ["llama emissions", "https://api.llama.fi/emissions"],
  ["coingecko list", "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=3&page=1"],
  ["tokenpost rss", "https://www.tokenpost.kr/rss"],
  ["theblock wp", "https://www.theblock.co/wp-json/wp/v2/posts?per_page=3"],
  ["decrypt wp", "https://decrypt.co/wp-json/wp/v2/posts?per_page=3"],
  ["coindesk sitemap", "https://www.coindesk.com/sitemap.xml"],
  ["cointelegraph sitemap", "https://cointelegraph.com/sitemap.xml"],
];
for (const [name, url] of urls) {
  try {
    const res = await fetch(url, { headers: { "User-Agent": UA, Accept: "application/json, text/html;q=0.8, */*;q=0.5" } });
    const text = await res.text();
    console.log(`\n### ${name} → ${res.status} (${text.length}자)\n${text.slice(0, 500).replace(/\s+/g, " ")}`);
  } catch (e) {
    console.log(`\n### ${name} → 실패 ${String(e?.cause?.code ?? e?.message ?? e)}`);
  }
}
