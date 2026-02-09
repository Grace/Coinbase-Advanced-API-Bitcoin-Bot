import crypto from "node:crypto";
import fs from "node:fs";

function usageAndExit() {
  console.error("Usage: node jwt_helper.mjs <key_file> <method> <path_with_query>");
  process.exit(2);
}

const args = process.argv.slice(2);
if (args.length !== 3) {
  usageAndExit();
}

const [keyFile, methodRaw, pathRaw] = args;
const method = methodRaw.toUpperCase();
const path = pathRaw.startsWith("/") ? pathRaw : `/${pathRaw}`;

let parsed;
try {
  parsed = JSON.parse(fs.readFileSync(keyFile, "utf8"));
} catch (err) {
  console.error(`Failed to load key file: ${err.message}`);
  process.exit(1);
}

const keyName = parsed?.name;
const privateKey = parsed?.privateKey;
if (!keyName || !privateKey) {
  console.error("Key file must contain fields: name, privateKey");
  process.exit(1);
}

const now = Math.floor(Date.now() / 1000);
const payload = {
  sub: keyName,
  iss: "cdp",
  nbf: now,
  exp: now + 120,
  uri: `${method} api.coinbase.com${path}`,
};

const headers = {
  alg: "ES256",
  typ: "JWT",
  kid: keyName,
  nonce: crypto.randomBytes(16).toString("hex"),
};

const encodedHeader = Buffer.from(JSON.stringify(headers)).toString("base64url");
const encodedPayload = Buffer.from(JSON.stringify(payload)).toString("base64url");
const signingInput = `${encodedHeader}.${encodedPayload}`;

let signature;
try {
  signature = crypto
    .createSign("SHA256")
    .update(signingInput)
    .end()
    .sign({ key: privateKey, dsaEncoding: "ieee-p1363" });
} catch (err) {
  console.error(`Failed to sign JWT: ${err.message}`);
  process.exit(1);
}

const encodedSignature = signature.toString("base64url");
process.stdout.write(`${signingInput}.${encodedSignature}`);
