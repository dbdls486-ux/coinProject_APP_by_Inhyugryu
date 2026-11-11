// 2차 시험: 코인니스·빗썸 공지·바이낸스 공지의 과거 페이지 넘기는 방법
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";
const get = async (url) => {
  try {
    const res = await fetch(url, { headers: { "User-Agent": UA, Accept: "application/json" } });
    return [res.status, await res.text()];
  } catch (e) {
    return [0, String(e?.cause?.code ?? e?.message ?? e)];
  }
};
const show = async (name, url, n = 700) => {
  const [s, t] = await get(url);
  console.log(`\n### ${name} → ${s} (${t.length}자)\n${t.slice(0, n).replace(/\s+/g, " ")}`);
  return t;
};
const C = "https://api.coinness.com/feed/v1";
const first = await show("coinness 1건 전체", `${C}/news?limit=1`, 3000);
for (const q of ["limit=3&lastId=1000000", "limit=3&cursor=1000000", "limit=3&page=500", "limit=3&offset=5000", "limit=3&id=1000000", "limit=3&maxId=1000000", "limit=3&before=1000000", "limit=3&publishAt=2021-05-01T00:00:00", "limit=3&endDate=2021-05-01", "limit=3&startDate=2021-05-01&endDate=2021-05-02", "limit=200", "limit=3&categoryId=4"]) {
  const [s, t] = await get(`${C}/news?${q}`);
  let ids = "";
  try {
    const j = JSON.parse(t);
    const arr = Array.isArray(j) ? j : j.data ?? j.items ?? j.list ?? [];
    ids = arr.slice(0, 4).map((x) => `${x.id}@${String(x.publishAt).slice(0, 16)}`).join(", ") + ` (총 ${arr.length}건)`;
  } catch {
    ids = t.slice(0, 200);
  }
  console.log(`coinness ?${q} → ${s} ${ids}`);
}
await show("coinness 단건", `${C}/news/1000000`, 600);
await show("coinness breaking", `${C}/breaking-news?limit=3&languageCode=ko`, 600);
await show("coinness breaking lastId", `${C}/breaking-news?limit=3&languageCode=ko&lastId=1000000`, 600);
await show("coinness v2 breaking", `https://api.coinness.com/feed/v2/breaking-news?limit=3&languageCode=ko`, 600);
await show("coinness categories", `${C}/categories`, 800);
await show("coinness news-categories", `${C}/news-categories`, 800);
// 빗썸 공지: 얼마나 과거까지 있나
for (const p of [1, 50, 200, 500, 1000]) {
  const [s, t] = await get(`https://api.bithumb.com/v1/notices?page=${p}&count=20`);
  let info = t.slice(0, 150);
  try {
    const j = JSON.parse(t);
    info = `${j.length}건 ${j[0]?.published_at} "${j[0]?.title}" [${j[0]?.categories}] ~ ${j.at(-1)?.published_at}`;
  } catch {}
  console.log(`bithumb notices page ${p} → ${s} ${info}`);
}
// 바이낸스 공지: 신규 상장(48), 상장폐지(161) 마지막 페이지
for (const [cat, page] of [[48, 1], [48, 40], [48, 115], [161, 1], [161, 20]]) {
  const [s, t] = await get(`https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&catalogId=${cat}&pageNo=${page}&pageSize=20`);
  let info = t.slice(0, 150);
  try {
    const c = JSON.parse(t).data.catalogs[0];
    info = `총 ${c.total} · ${c.articles.length}건 · ${new Date(c.articles[0].releaseDate).toISOString().slice(0, 10)} "${c.articles[0].title}" ~ ${new Date(c.articles.at(-1).releaseDate).toISOString().slice(0, 10)}`;
  } catch {}
  console.log(`binance catalog ${cat} page ${page} → ${s} ${info}`);
}
// 블록미디어: 총 글 수와 페이지당 한도
{
  const res = await fetch("https://www.blockmedia.co.kr/wp-json/wp/v2/posts?per_page=100&_fields=id,date_gmt,title&before=2022-01-01T00:00:00", { headers: { "User-Agent": UA } });
  console.log(`blockmedia per_page=100 → ${res.status} total ${res.headers.get("x-wp-total")} pages ${res.headers.get("x-wp-totalpages")}`);
}
