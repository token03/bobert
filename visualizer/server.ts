const server = Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") return new Response(Bun.file("index.html"));
    if (url.pathname === "/style.css") return new Response(Bun.file("style.css"));
    
    if (url.pathname.startsWith("/viz_data/")) {
      const fileName = url.pathname.replace("/viz_data/", "");
      return new Response(Bun.file(`viz_data/${fileName}`));
    }

    if (url.pathname === "/index.js") {
        const build = await Bun.build({
            entrypoints: ["./src/index.ts"],
            minify: false, 
        });
        return new Response(build.outputs[0]);
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Listening on http://localhost:${server.port}`);