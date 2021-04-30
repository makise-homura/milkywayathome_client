#include "crlibm.h"
#include "crlibm_private.h"

/*File generated by maple/exp-td.mpl*/
#define L 12
#define LHALF 6
#define log2InvMult2L 5.90927888748119403317105025053024291992187500000000e+03
#define msLog2Div2Lh -1.69225385878892892145206050535932718048570677638054e-04
#define msLog2Div2Lm -5.66173538536694228129338642508144915789549786200850e-21
#define msLog2Div2Ll -1.39348350547270802387134479195513221596886344528448e-37
#define shiftConst 6.75539944105574400000000000000000000000000000000000e+15
#define INDEXMASK1 0x0000003f
#define INDEXMASK2 0x00000fc0
#define OVRUDRFLWSMPLBOUND 0x4086232b
#define OVRFLWBOUND 7.09782712893383973096206318587064743041992187500000e+02
#define LARGEST 1.79769313486231570814527423731704356798070567525845e+308
#define SMALLEST 4.94065645841246544176568792868221372365059802614325e-324
#define DENORMBOUND -7.08396418532264078748994506895542144775390625000000e+02
#define UNDERFLWBOUND -7.45133219101941222106688655912876129150390625000000e+02
#define twoPowerM1000 9.33263618503218878990089544723817169617091446371708e-302
#define twoPower1000 1.07150860718626732094842504906000181056140481170553e+301
#define ROUNDCST 1.00097751710654947574476244840271277078237100388308e+00
#define RDROUNDCST 5.42101086242752217003726400434970855712890625000000e-20
#define twoM52 2.22044604925031308084726333618164062500000000000000e-16
#define mTwoM53 -1.11022302462515654042363166809082031250000000000000e-16


#define c3 1.66666666696497128841158996692684013396501541137695e-01
#define c4 4.16666666766101478902584176466916687786579132080078e-02
#define accPolyC3h 1.66666666666666657414808128123695496469736099243164e-01
#define accPolyC3l 9.25185853854303382711778796549658869198239876296876e-18
#define accPolyC4h 4.16666666666666643537020320309238741174340248107910e-02
#define accPolyC4l 2.31256737763466047834406500740034679807274635559840e-18
#define accPolyC5 8.33333333333333321768510160154619370587170124053955e-03
#define accPolyC6 1.38888888911084160647513296993338371976278722286224e-03
#define accPolyC7 1.98412698447204637443655461659375305316643789410591e-04


typedef struct tPi_t_tag {double hi; double mi; double lo;} tPi_t;
static const tPi_t twoPowerIndex1[64] = {
  {
    1.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex1hi[0] */
    0.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex1mi[0] */
    0.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex1lo[0] */
  } ,
  {
    1.00016923970530213772178740327944979071617126464844e+00, /* twoPowerIndex1hi[1] */
    9.33618533547846199388179470637929281332443150415515e-17, /* twoPowerIndex1mi[1] */
    3.77297954885090084710073756456287936644477196404411e-33, /* twoPowerIndex1lo[1] */
  } ,
  {
    1.00033850805268231809463941317517310380935668945312e+00, /* twoPowerIndex1hi[2] */
    -5.14133393131895706530678267769194263660821897510018e-18, /* twoPowerIndex1mi[2] */
    3.65532984508913994275805274025238001612209382982528e-34, /* twoPowerIndex1lo[2] */
  } ,
  {
    1.00050780504698755279946453811135143041610717773438e+00, /* twoPowerIndex1hi[3] */
    6.96242402202057255168575283120383384543281723609229e-17, /* twoPowerIndex1mi[3] */
    5.20646484360987194687938744912769762301408608274186e-34, /* twoPowerIndex1lo[3] */
  } ,
  {
    1.00067713069306640782940576173132285475730895996094e+00, /* twoPowerIndex1hi[4] */
    -5.11512329768566763581175858027414623025077801052578e-17, /* twoPowerIndex1mi[4] */
    6.98198098827029289611306311409722598417652045670746e-34, /* twoPowerIndex1lo[4] */
  } ,
  {
    1.00084648499576744917760606767842546105384826660156e+00, /* twoPowerIndex1hi[5] */
    8.42299002458648658027371639395540881533737558365359e-17, /* twoPowerIndex1mi[5] */
    2.36970468083963275494383447607242933566525306383712e-33, /* twoPowerIndex1lo[5] */
  } ,
  {
    1.00101586795994101919404783984646201133728027343750e+00, /* twoPowerIndex1hi[6] */
    -2.82452207477616780689379931990767172214919031737312e-17, /* twoPowerIndex1mi[6] */
    -1.95728551891328398835983018867545661396300096090192e-33, /* twoPowerIndex1lo[6] */
  } ,
  {
    1.00118527959043746022871346212923526763916015625000e+00, /* twoPowerIndex1hi[7] */
    -7.18042456559213166577815896739695813826899291527385e-17, /* twoPowerIndex1mi[7] */
    -4.76259460142737667571053552520587169925041399888400e-33, /* twoPowerIndex1lo[7] */
  } ,
  {
    1.00135471989210822485460994357708841562271118164062e+00, /* twoPowerIndex1hi[8] */
    -1.89737284167929933536294189134998232481318340155287e-17, /* twoPowerIndex1mi[8] */
    -1.14644761608474571162584330231223616391499645510675e-34, /* twoPowerIndex1lo[8] */
  } ,
  {
    1.00152418886980565382316399336559697985649108886719e+00, /* twoPowerIndex1hi[9] */
    9.06044106726912172835402556659442028244667434824268e-17, /* twoPowerIndex1mi[9] */
    -9.28795936877474030548429956901821606468166518236870e-34, /* twoPowerIndex1lo[9] */
  } ,
  {
    1.00169368652838319810882694582687690854072570800781e+00, /* twoPowerIndex1hi[10] */
    -7.17327634990031997530849131951948428542423257707819e-17, /* twoPowerIndex1mi[10] */
    6.07647560689104755979568262152596670382288431776872e-33, /* twoPowerIndex1lo[10] */
  } ,
  {
    1.00186321287269430868605013529304414987564086914062e+00, /* twoPowerIndex1hi[11] */
    -1.33071962467226606689858482393149015112554043251841e-17, /* twoPowerIndex1mi[11] */
    -3.85443386097781554341108465500150047004226523597749e-34, /* twoPowerIndex1lo[11] */
  } ,
  {
    1.00203276790759399084151937131537124514579772949219e+00, /* twoPowerIndex1hi[12] */
    2.57269259432211179649018725316354562817431777421067e-17, /* twoPowerIndex1mi[12] */
    6.79751255613789265201334995370535043187556454092516e-34, /* twoPowerIndex1lo[12] */
  } ,
  {
    1.00220235163793791599573523853905498981475830078125e+00, /* twoPowerIndex1hi[13] */
    -3.92993778548451720552933864302152846562830407163222e-17, /* twoPowerIndex1mi[13] */
    2.22740945016647607963332576015739728975193247400334e-33, /* twoPowerIndex1lo[13] */
  } ,
  {
    1.00237196406858219965840817167190834879875183105469e+00, /* twoPowerIndex1hi[14] */
    8.46137724799471749932697856411684097253648908880386e-17, /* twoPowerIndex1mi[14] */
    -1.15795350476069740873680647805085951983720786847012e-33, /* twoPowerIndex1lo[14] */
  } ,
  {
    1.00254160520438451165148308064090088009834289550781e+00, /* twoPowerIndex1hi[15] */
    -4.19488324163994025318330321149973224973566383682287e-17, /* twoPowerIndex1mi[15] */
    -1.10867727371575753136248480661711257993643946476249e-33, /* twoPowerIndex1lo[15] */
  } ,
  {
    1.00271127505020252179690487537300214171409606933594e+00, /* twoPowerIndex1hi[16] */
    -3.63661592869226394431650732319448247819940912156384e-17, /* twoPowerIndex1mi[16] */
    -6.58076018413558386665763067745208065096307047224855e-34, /* twoPowerIndex1lo[16] */
  } ,
  {
    1.00288097361089523218424801598303020000457763671875e+00, /* twoPowerIndex1hi[17] */
    -2.61094406324393831074545742903203024867557691172893e-17, /* twoPowerIndex1mi[17] */
    1.20560913995705683042979390390212408639703633013464e-33, /* twoPowerIndex1lo[17] */
  } ,
  {
    1.00305070089132231103690173767972737550735473632812e+00, /* twoPowerIndex1hi[18] */
    1.75307847798233211014008193947934807324617806015147e-17, /* twoPowerIndex1mi[18] */
    1.13393213893456062969952574781413432615748154285102e-33, /* twoPowerIndex1lo[18] */
  } ,
  {
    1.00322045689634431475667497579706832766532897949219e+00, /* twoPowerIndex1hi[19] */
    5.75392352562826744411122442879351687879645730521252e-17, /* twoPowerIndex1mi[19] */
    -2.67723202763627633853056570968049475423332897231238e-33, /* twoPowerIndex1lo[19] */
  } ,
  {
    1.00339024163082268792379636579426005482673645019531e+00, /* twoPowerIndex1hi[20] */
    -8.68492200511795617368964392061338881522209699276521e-18, /* twoPowerIndex1mi[20] */
    -3.48402050975972235345434529928044249018150775634918e-34, /* twoPowerIndex1lo[20] */
  } ,
  {
    1.00356005509961931920770439319312572479248046875000e+00, /* twoPowerIndex1hi[21] */
    9.49003543098177759384527648244793376993805358019830e-17, /* twoPowerIndex1mi[21] */
    -2.72838479693459377473949067074824993718971708464416e-34, /* twoPowerIndex1lo[21] */
  } ,
  {
    1.00372989730759765159007201873464509844779968261719e+00, /* twoPowerIndex1hi[22] */
    -8.71038060581842237129503222771273097866222639659196e-17, /* twoPowerIndex1mi[22] */
    1.08618280977421918449964280592134822915254002216492e-33, /* twoPowerIndex1lo[22] */
  } ,
  {
    1.00389976825962090600796727812848985195159912109375e+00, /* twoPowerIndex1hi[23] */
    3.49589169585715450145113080496294928974704413200507e-17, /* twoPowerIndex1mi[23] */
    -2.94144759791575889018758341844218078845293410867033e-33, /* twoPowerIndex1lo[23] */
  } ,
  {
    1.00406966796055407975529760733479633927345275878906e+00, /* twoPowerIndex1hi[24] */
    9.75378754984024099793268477195846557070899843396028e-17, /* twoPowerIndex1mi[24] */
    5.64300142720863839874652371486135387544215443970822e-33, /* twoPowerIndex1lo[24] */
  } ,
  {
    1.00423959641526283625978521740762516856193542480469e+00, /* twoPowerIndex1hi[25] */
    -1.05762211962928569325595987354393790031696252155330e-16, /* twoPowerIndex1mi[25] */
    5.83707943855262433756426258532402998630656838432942e-33, /* twoPowerIndex1lo[25] */
  } ,
  {
    1.00440955362861283894915231940103694796562194824219e+00, /* twoPowerIndex1hi[26] */
    4.20918873812712592966120368574870018412444102186114e-17, /* twoPowerIndex1mi[26] */
    2.17971061415864797757688538084764343153482363480001e-33, /* twoPowerIndex1lo[26] */
  } ,
  {
    1.00457953960547174965256544965086504817008972167969e+00, /* twoPowerIndex1hi[27] */
    -1.67001668575547876279593068362269006396955075031486e-17, /* twoPowerIndex1mi[27] */
    8.56552273127180632250100064178508499726804497630949e-34, /* twoPowerIndex1lo[27] */
  } ,
  {
    1.00474955435070723019919114449294283986091613769531e+00, /* twoPowerIndex1hi[28] */
    -1.62314635541245143576050143734168865049382925451905e-17, /* twoPowerIndex1mi[28] */
    9.28530603331304279248127378347521992965436217888846e-35, /* twoPowerIndex1lo[28] */
  } ,
  {
    1.00491959786918805264122056541964411735534667968750e+00, /* twoPowerIndex1hi[29] */
    2.30285392780281171183781407072193459566423534757738e-17, /* twoPowerIndex1mi[29] */
    -4.94345170697772961451844215454960159293576885861471e-34, /* twoPowerIndex1lo[29] */
  } ,
  {
    1.00508967016578387720926457404857501387596130371094e+00, /* twoPowerIndex1hi[30] */
    1.64180469767730323539044065876968510741128671646342e-17, /* twoPowerIndex1mi[30] */
    -1.40120338097898451052996285664279342840562084008057e-34, /* twoPowerIndex1lo[30] */
  } ,
  {
    1.00525977124536503026774880709126591682434082031250e+00, /* twoPowerIndex1hi[31] */
    3.72669843182841367415839881731427710949489409371967e-17, /* twoPowerIndex1mi[31] */
    -6.47401139805376556406832921347318553328966160972009e-34, /* twoPowerIndex1lo[31] */
  } ,
  {
    1.00542990111280272635951860138447955250740051269531e+00, /* twoPowerIndex1hi[32] */
    9.49918653545503175702165599813853704096085691656846e-17, /* twoPowerIndex1mi[32] */
    2.69197614795285565383714012654266434015814803945559e-33, /* twoPowerIndex1lo[32] */
  } ,
  {
    1.00560005977296929025044391892151907086372375488281e+00, /* twoPowerIndex1hi[33] */
    -8.68093131444458156819420437418787120833640771986467e-17, /* twoPowerIndex1mi[33] */
    1.70120132411898592857823454488038166970266621441155e-33, /* twoPowerIndex1lo[33] */
  } ,
  {
    1.00577024723073704670639472169568762183189392089844e+00, /* twoPowerIndex1hi[34] */
    4.00054749103011688253297776917861866454998331259826e-17, /* twoPowerIndex1mi[34] */
    1.31925963209617013098908877209722537258440391267352e-33, /* twoPowerIndex1lo[34] */
  } ,
  {
    1.00594046349098009685008037195075303316116333007812e+00, /* twoPowerIndex1hi[35] */
    7.19049911150997399545071999536183023264065701084941e-17, /* twoPowerIndex1mi[35] */
    5.87388840736458866162752231275781178461761843642826e-33, /* twoPowerIndex1lo[35] */
  } ,
  {
    1.00611070855857298589342008199309930205345153808594e+00, /* twoPowerIndex1hi[36] */
    -1.39080686710657830063564897325456873924612265621065e-17, /* twoPowerIndex1mi[36] */
    -1.04608842475341479042763417006066485614641577765254e-33, /* twoPowerIndex1lo[36] */
  } ,
  {
    1.00628098243839092518214783922303467988967895507812e+00, /* twoPowerIndex1hi[37] */
    -8.14020864257304964668281784838729787981918541118695e-17, /* twoPowerIndex1mi[37] */
    9.81687621053626949463979101287162593669867055078380e-34, /* twoPowerIndex1lo[37] */
  } ,
  {
    1.00645128513531001424041733116609975695610046386719e+00, /* twoPowerIndex1hi[38] */
    -5.76215104374953424938341851527793234314721266620465e-17, /* twoPowerIndex1mi[38] */
    -2.57074696660737495074482007072094547944536206743811e-33, /* twoPowerIndex1lo[38] */
  } ,
  {
    1.00662161665420724077080194547306746244430541992188e+00, /* twoPowerIndex1hi[39] */
    6.74527847731045679518854873240614524466814727371618e-17, /* twoPowerIndex1mi[39] */
    3.78807035520293397050945635744099720111701312971531e-33, /* twoPowerIndex1lo[39] */
  } ,
  {
    1.00679197699996070269889969495125114917755126953125e+00, /* twoPowerIndex1hi[40] */
    1.89985572403462958330039110072328316553950667129559e-17, /* twoPowerIndex1mi[40] */
    -9.63431595756658515104002497950853752466876076619761e-34, /* twoPowerIndex1lo[40] */
  } ,
  {
    1.00696236617744894203951844247058033943176269531250e+00, /* twoPowerIndex1hi[41] */
    -9.63743003231640715845365699595418727819327585110316e-17, /* twoPowerIndex1mi[41] */
    5.37739172125896523173793245498965828230574349410153e-34, /* twoPowerIndex1lo[41] */
  } ,
  {
    1.00713278419155116694128082599490880966186523437500e+00, /* twoPowerIndex1hi[42] */
    -1.25286544624539788678561179100718466970708089928402e-17, /* twoPowerIndex1mi[42] */
    6.73461401738595970431440883296934068477066941591536e-34, /* twoPowerIndex1lo[42] */
  } ,
  {
    1.00730323104714791782043903367593884468078613281250e+00, /* twoPowerIndex1hi[43] */
    3.02057888784369419361601396858156577301726732087163e-17, /* twoPowerIndex1mi[43] */
    2.28748779695564659338938424768062274005093770790899e-33, /* twoPowerIndex1lo[43] */
  } ,
  {
    1.00747370674912040122706002875929698348045349121094e+00, /* twoPowerIndex1hi[44] */
    -4.86939425860856490657306074829518805252629724182687e-17, /* twoPowerIndex1mi[44] */
    -2.15005825118530979132585793750524459885823808259600e-33, /* twoPowerIndex1lo[44] */
  } ,
  {
    1.00764421130235026780042062455322593450546264648438e+00, /* twoPowerIndex1hi[45] */
    5.22402993768745316546273304631524097955173932918745e-17, /* twoPowerIndex1mi[45] */
    1.56859855755260687389551771821040360984576895950448e-33, /* twoPowerIndex1lo[45] */
  } ,
  {
    1.00781474471172072249203210958512499928474426269531e+00, /* twoPowerIndex1hi[46] */
    -9.36154355147845590745647945463041972879607082944076e-17, /* twoPowerIndex1mi[46] */
    2.73525232969860711971224663796949142049450209755900e-33, /* twoPowerIndex1lo[46] */
  } ,
  {
    1.00798530698211497025340577238239347934722900390625e+00, /* twoPowerIndex1hi[47] */
    -8.65251323306194956904999437269495581443314540350323e-17, /* twoPowerIndex1mi[47] */
    3.95202635576858744637745517221782298786195221857564e-33, /* twoPowerIndex1lo[47] */
  } ,
  {
    1.00815589811841754830368245166027918457984924316406e+00, /* twoPowerIndex1hi[48] */
    -3.25205875608430806088583499076669170064325943797327e-17, /* twoPowerIndex1mi[48] */
    2.46355206137317856028163445564386882918089954604456e-33, /* twoPowerIndex1lo[48] */
  } ,
  {
    1.00832651812551388204042268625926226377487182617188e+00, /* twoPowerIndex1hi[49] */
    -9.91723226806091428304981281566311898208154515630026e-17, /* twoPowerIndex1mi[49] */
    -5.54888559019883709197982155449053331751526019176611e-33, /* twoPowerIndex1lo[49] */
  } ,
  {
    1.00849716700828984095039686508243903517723083496094e+00, /* twoPowerIndex1hi[50] */
    -7.13604740416252276802131825807759910735798752142063e-17, /* twoPowerIndex1mi[50] */
    4.98539917129953776836361552714923793985513981224277e-33, /* twoPowerIndex1lo[50] */
  } ,
  {
    1.00866784477163240474340000218944624066352844238281e+00, /* twoPowerIndex1hi[51] */
    -1.72686837122432199045769102324975071816159176955755e-17, /* twoPowerIndex1mi[51] */
    1.49699250027049460850618457278328039699137666243513e-33, /* twoPowerIndex1lo[51] */
  } ,
  {
    1.00883855142042944130764681176515296101570129394531e+00, /* twoPowerIndex1hi[52] */
    -6.61995469367394011396161169543794432105455894234207e-17, /* twoPowerIndex1mi[52] */
    -1.37180394060118324236386341775148249159351409372011e-33, /* twoPowerIndex1lo[52] */
  } ,
  {
    1.00900928695956926262056185805704444646835327148438e+00, /* twoPowerIndex1hi[53] */
    3.56545690151302037529536411250390721695546928174132e-17, /* twoPowerIndex1mi[53] */
    -3.04973969952342772070810129193852120178103073824881e-33, /* twoPowerIndex1lo[53] */
  } ,
  {
    1.00918005139394151292719925550045445561408996582031e+00, /* twoPowerIndex1hi[54] */
    3.71731001370881785990046067286312459291233468170260e-17, /* twoPowerIndex1mi[54] */
    2.33846700487585833881864340821781209486455210644683e-33, /* twoPowerIndex1lo[54] */
  } ,
  {
    1.00935084472843628056182296859333291649818420410156e+00, /* twoPowerIndex1hi[55] */
    7.06257240682552768409771521622687712943771198652841e-17, /* twoPowerIndex1mi[55] */
    -2.69254581845265407066861039619808678678301305791490e-33, /* twoPowerIndex1lo[55] */
  } ,
  {
    1.00952166696794476408172158699017018079757690429688e+00, /* twoPowerIndex1hi[56] */
    -1.43214123034288192889278157016818203958577074101904e-17, /* twoPowerIndex1mi[56] */
    1.18343254028162625925516501493235924385911429505127e-33, /* twoPowerIndex1lo[56] */
  } ,
  {
    1.00969251811735860613339355040807276964187622070312e+00, /* twoPowerIndex1hi[57] */
    1.56681880131341096294393603553275561732689802958880e-17, /* twoPowerIndex1mi[57] */
    -7.81531085085694011629938080711347719818453468652371e-34, /* twoPowerIndex1lo[57] */
  } ,
  {
    1.00986339818157078163096684875199571251869201660156e+00, /* twoPowerIndex1hi[58] */
    -1.10436957803936884179513517801130489140214758906396e-16, /* twoPowerIndex1mi[58] */
    2.26009746193166539328485913376118607218426838997178e-33, /* twoPowerIndex1lo[58] */
  } ,
  {
    1.01003430716547448753317439695820212364196777343750e+00, /* twoPowerIndex1hi[59] */
    -5.76731742716039801926309676866051407637690838590278e-17, /* twoPowerIndex1mi[59] */
    -2.78217177127226696492717790709599913537289113429847e-33, /* twoPowerIndex1lo[59] */
  } ,
  {
    1.01020524507396425306637866015080362558364868164062e+00, /* twoPowerIndex1hi[60] */
    4.83548497844038273903057565303994337100835235326315e-18, /* twoPowerIndex1mi[60] */
    2.61454310671247390892899439237254056446677108734060e-34, /* twoPowerIndex1lo[60] */
  } ,
  {
    1.01037621191193527359075687854783609509468078613281e+00, /* twoPowerIndex1hi[61] */
    7.01512128971544209746870624992521939688820868468409e-17, /* twoPowerIndex1mi[61] */
    -5.58161827166019262256128049263563333632096890407031e-33, /* twoPowerIndex1lo[61] */
  } ,
  {
    1.01054720768428363264490599249256774783134460449219e+00, /* twoPowerIndex1hi[62] */
    7.16180287361957384315239192590148379997244337618299e-17, /* twoPowerIndex1mi[62] */
    -2.02563561553844543425647289447795720504633264470840e-34, /* twoPowerIndex1lo[62] */
  } ,
  {
    1.01071823239590607990123771742219105362892150878906e+00, /* twoPowerIndex1hi[63] */
    1.05046591340840499506734116899577087874840087102554e-16, /* twoPowerIndex1mi[63] */
    2.18415845005431430458109722226729544983351542501345e-33, /* twoPowerIndex1lo[63] */
  } };

static const tPi_t twoPowerIndex2[64] = {
  {
    1.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex2hi[0] */
    0.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex2mi[0] */
    0.00000000000000000000000000000000000000000000000000e+00, /* twoPowerIndex2lo[0] */
  } ,
  {
    1.01088928605170047525518839393043890595436096191406e+00, /* twoPowerIndex2hi[1] */
    -1.52347786033685771762884896542837070154735103108242e-17, /* twoPowerIndex2mi[1] */
    -1.20527773363982029579667170059568717373235495431222e-33, /* twoPowerIndex2lo[1] */
  } ,
  {
    1.02189714865411662714222984504885971546173095703125e+00, /* twoPowerIndex2hi[2] */
    5.10922502897344389358977529484899213711118891193073e-17, /* twoPowerIndex2mi[2] */
    7.88422656496927442160138555121324022911581088833303e-34, /* twoPowerIndex2lo[2] */
  } ,
  {
    1.03302487902122841489926940994337201118469238281250e+00, /* twoPowerIndex2hi[3] */
    7.60083887402708848935418174694583720224989620170520e-18, /* twoPowerIndex2mi[3] */
    4.17547660336499599627565312970423748685953776150432e-34, /* twoPowerIndex2lo[3] */
  } ,
  {
    1.04427378242741375480306942336028441786766052246094e+00, /* twoPowerIndex2hi[4] */
    8.55188970553796489217080231819249267647451456689049e-17, /* twoPowerIndex2mi[4] */
    -4.33079108057472301914865017639792677602990877842667e-33, /* twoPowerIndex2lo[4] */
  } ,
  {
    1.05564517836055715704901558638084679841995239257812e+00, /* twoPowerIndex2hi[5] */
    1.75932573877209198413667791203269037000656546957040e-18, /* twoPowerIndex2mi[5] */
    -1.30396724977978376996223394717384715710429730879060e-34, /* twoPowerIndex2lo[5] */
  } ,
  {
    1.06714040067682369716806078940862789750099182128906e+00, /* twoPowerIndex2hi[6] */
    -7.89985396684158212226333428391940791349522867645719e-17, /* twoPowerIndex2mi[6] */
    2.48773924323047906697979838020484860096229450391065e-33, /* twoPowerIndex2lo[6] */
  } ,
  {
    1.07876079775711986030728439800441265106201171875000e+00, /* twoPowerIndex2hi[7] */
    -6.65666043605659260344452997720816205593857378631714e-17, /* twoPowerIndex2mi[7] */
    -3.65812580131923690564246115619834327696213789275798e-33, /* twoPowerIndex2lo[7] */
  } ,
  {
    1.09050773266525768967483145388541743159294128417969e+00, /* twoPowerIndex2hi[8] */
    -3.04678207981247114696551170541257032193091359539676e-17, /* twoPowerIndex2mi[8] */
    2.01705487848848619150139275849815038703452970938356e-33, /* twoPowerIndex2lo[8] */
  } ,
  {
    1.10238258330784089089604549371870234608650207519531e+00, /* twoPowerIndex2hi[9] */
    5.26603687157069438656191942579725322612448063343618e-17, /* twoPowerIndex2mi[9] */
    6.45805397536721410708490440561131106837447579889036e-34, /* twoPowerIndex2lo[9] */
  } ,
  {
    1.11438674259589243220602838846389204263687133789062e+00, /* twoPowerIndex2hi[10] */
    1.04102784568455709549461912557590544266030834754874e-16, /* twoPowerIndex2mi[10] */
    1.47570167344000313963428137759389241586071635800536e-33, /* twoPowerIndex2lo[10] */
  } ,
  {
    1.12652161860824184813623105583246797323226928710938e+00, /* twoPowerIndex2hi[11] */
    5.16585675879545673703180814711785602957687143067887e-17, /* twoPowerIndex2mi[11] */
    -5.65916686170716220063974592097473710424342945964456e-34, /* twoPowerIndex2lo[11] */
  } ,
  {
    1.13878863475669156457570352358743548393249511718750e+00, /* twoPowerIndex2hi[12] */
    8.91281267602540777782023808215740339330966556723006e-17, /* twoPowerIndex2mi[12] */
    -2.00741463283249448727846279877956608996968961458494e-33, /* twoPowerIndex2lo[12] */
  } ,
  {
    1.15118922995298267331065744656370952725410461425781e+00, /* twoPowerIndex2hi[13] */
    3.25071021886382721197741051783883831627922464835734e-17, /* twoPowerIndex2mi[13] */
    8.89091931637927159755660842958283684863470887791663e-34, /* twoPowerIndex2lo[13] */
  } ,
  {
    1.16372485877757747552152522985124960541725158691406e+00, /* twoPowerIndex2hi[14] */
    3.82920483692409349872159816102110059722238312159091e-17, /* twoPowerIndex2mi[14] */
    7.19709831987676327409679245188515478735702972651365e-34, /* twoPowerIndex2lo[14] */
  } ,
  {
    1.17639699165028122074261318630306050181388854980469e+00, /* twoPowerIndex2hi[15] */
    5.55420325421807896276684328889766000203735373777665e-17, /* twoPowerIndex2mi[15] */
    -1.48842929343368511961710849782815827921712348302943e-33, /* twoPowerIndex2lo[15] */
  } ,
  {
    1.18920711500272102689734765590401366353034973144531e+00, /* twoPowerIndex2hi[16] */
    3.98201523146564611098029654755651628169309930562261e-17, /* twoPowerIndex2mi[16] */
    1.14195965688545340101163692105448367710482873511510e-33, /* twoPowerIndex2lo[16] */
  } ,
  {
    1.20215673145270307564658196497475728392601013183594e+00, /* twoPowerIndex2hi[17] */
    6.64498149925230124489270286991048474954122885069094e-17, /* twoPowerIndex2mi[17] */
    -3.85685255336907654452995968372311171004938395576078e-33, /* twoPowerIndex2lo[17] */
  } ,
  {
    1.21524735998046895524282717815367504954338073730469e+00, /* twoPowerIndex2hi[18] */
    -7.71263069268148813091257203929615917892526180720328e-17, /* twoPowerIndex2mi[18] */
    4.71720614288499816545262331753743129502653902881319e-33, /* twoPowerIndex2lo[18] */
  } ,
  {
    1.22848053610687002468182527081808075308799743652344e+00, /* twoPowerIndex2hi[19] */
    -1.89878163130252995311948719658078062677878185721429e-17, /* twoPowerIndex2mi[19] */
    6.18469453652103848424687212656185051850860290411013e-34, /* twoPowerIndex2lo[19] */
  } ,
  {
    1.24185781207348400201340155035723000764846801757812e+00, /* twoPowerIndex2hi[20] */
    4.65802759183693679122616767654141683800114091774588e-17, /* twoPowerIndex2mi[20] */
    -2.31439910378785985764079190411324307370918857557275e-33, /* twoPowerIndex2lo[20] */
  } ,
  {
    1.25538075702469109629078047873917967081069946289062e+00, /* twoPowerIndex2hi[21] */
    -6.71138982129687841852821870522008036246320680675952e-18, /* twoPowerIndex2mi[21] */
    -5.76846264325028352862497671463784812858487529613818e-35, /* twoPowerIndex2lo[21] */
  } ,
  {
    1.26905095719173321988648694969015195965766906738281e+00, /* twoPowerIndex2hi[22] */
    2.66793213134218609522998616502989391061315227839969e-18, /* twoPowerIndex2mi[22] */
    -5.01723570938719050333027020376949013774930719140568e-35, /* twoPowerIndex2lo[22] */
  } ,
  {
    1.28287001607877826359072059858590364456176757812500e+00, /* twoPowerIndex2hi[23] */
    1.71359491824356096814175768900864730928374054987756e-17, /* twoPowerIndex2mi[23] */
    7.25131491282819461838977871983760669047046080773478e-34, /* twoPowerIndex2lo[23] */
  } ,
  {
    1.29683955465100964055125132290413603186607360839844e+00, /* twoPowerIndex2hi[24] */
    2.53825027948883149592910250791940344234383392801619e-17, /* twoPowerIndex2mi[24] */
    1.68678246461832500334243039646153124282288438904092e-34, /* twoPowerIndex2lo[24] */
  } ,
  {
    1.31096121152476441373835314152529463171958923339844e+00, /* twoPowerIndex2hi[25] */
    -7.18153613551945385697245613605196258733040544776187e-17, /* twoPowerIndex2mi[25] */
    -2.12629266743969557140434977160228094057588798528197e-34, /* twoPowerIndex2lo[25] */
  } ,
  {
    1.32523664315974132321684919588733464479446411132812e+00, /* twoPowerIndex2hi[26] */
    -2.85873121003886137327027220806882812126373511580572e-17, /* twoPowerIndex2mi[26] */
    7.62021406397260431456821182703024347950927217409368e-34, /* twoPowerIndex2lo[26] */
  } ,
  {
    1.33966752405330291608720472140703350305557250976562e+00, /* twoPowerIndex2hi[27] */
    8.92728259483173198426255486589831591472085466011828e-17, /* twoPowerIndex2mi[27] */
    -7.69657983531899254540849298389093495155758992415150e-34, /* twoPowerIndex2lo[27] */
  } ,
  {
    1.35425554693689265128853094211081042885780334472656e+00, /* twoPowerIndex2hi[28] */
    7.70094837980298946162338224151128414915778826123523e-17, /* twoPowerIndex2mi[28] */
    -2.24074836437395028882100810844688282941385619430371e-33, /* twoPowerIndex2lo[28] */
  } ,
  {
    1.36900242297459051599162194179370999336242675781250e+00, /* twoPowerIndex2hi[29] */
    9.59379791911884877255545693637832999511204427845953e-17, /* twoPowerIndex2mi[29] */
    -4.88674958784947176959607858061848292445663702071206e-33, /* twoPowerIndex2lo[29] */
  } ,
  {
    1.38390988196383202257777611521305516362190246582031e+00, /* twoPowerIndex2hi[30] */
    -6.77051165879478628715737183479431151106043475381389e-17, /* twoPowerIndex2mi[30] */
    5.25954134785524271676320971772030766300829615286373e-34, /* twoPowerIndex2lo[30] */
  } ,
  {
    1.39897967253831123635166022722842171788215637207031e+00, /* twoPowerIndex2hi[31] */
    -9.61421320905132307233280072508933760749140842467170e-17, /* twoPowerIndex2mi[31] */
    3.97465190077505680357285219692652769985214754647576e-33, /* twoPowerIndex2lo[31] */
  } ,
  {
    1.41421356237309514547462185873882845044136047363281e+00, /* twoPowerIndex2hi[32] */
    -9.66729331345291345105469972976694765012981670542977e-17, /* twoPowerIndex2mi[32] */
    4.13867530869941356271900493210877889450985709540127e-33, /* twoPowerIndex2lo[32] */
  } ,
  {
    1.42961333839197002326670826732879504561424255371094e+00, /* twoPowerIndex2hi[33] */
    -1.20316424890536551791763281075597751007148682598677e-17, /* twoPowerIndex2mi[33] */
    3.96492532243389364766543780399018506300743370884771e-35, /* twoPowerIndex2lo[33] */
  } ,
  {
    1.44518080697704665027458759141154587268829345703125e+00, /* twoPowerIndex2hi[34] */
    -3.02375813499398731939978948265280760393682335269040e-17, /* twoPowerIndex2mi[34] */
    -1.77301195820250091791088617662298487007284882395542e-33, /* twoPowerIndex2lo[34] */
  } ,
  {
    1.46091779418064704465507475106278434395790100097656e+00, /* twoPowerIndex2hi[35] */
    -5.60037718607521580013156831807759453639536208267684e-17, /* twoPowerIndex2mi[35] */
    -4.80948804890004400970317146361816382779746350931714e-33, /* twoPowerIndex2lo[35] */
  } ,
  {
    1.47682614593949934622685304930200800299644470214844e+00, /* twoPowerIndex2hi[36] */
    -3.48399455689279579579151031868718147769491495422105e-17, /* twoPowerIndex2mi[36] */
    -1.21157704523090580028169713170353136148082368354920e-34, /* twoPowerIndex2lo[36] */
  } ,
  {
    1.49290772829126483500772337720263749361038208007812e+00, /* twoPowerIndex2hi[37] */
    1.41929201542840357707378184476885202767753055101956e-17, /* twoPowerIndex2mi[37] */
    2.77326329344780505247628358108799734324783290873618e-34, /* twoPowerIndex2lo[37] */
  } ,
  {
    1.50916442759342284141155232646269723773002624511719e+00, /* twoPowerIndex2hi[38] */
    -1.01645532775429503910501990740249618370059871055172e-16, /* twoPowerIndex2mi[38] */
    2.04191706967403438352422808603561166583202022922508e-34, /* twoPowerIndex2lo[38] */
  } ,
  {
    1.52559815074453841710067081294255331158638000488281e+00, /* twoPowerIndex2hi[39] */
    -1.10249417123425609363148008789604625195179292613569e-16, /* twoPowerIndex2mi[39] */
    -2.99382882637137806007903782085057425945683820190483e-33, /* twoPowerIndex2lo[39] */
  } ,
  {
    1.54221082540794074411394376511452719569206237792969e+00, /* twoPowerIndex2hi[40] */
    7.94983480969762085616103882937991564856794389991833e-17, /* twoPowerIndex2mi[40] */
    -9.15995637410036729585390444224530830478731117122757e-34, /* twoPowerIndex2lo[40] */
  } ,
  {
    1.55900440023783692922165755589958280324935913085938e+00, /* twoPowerIndex2hi[41] */
    3.78120705335752750188190562589679090842557793649900e-17, /* twoPowerIndex2mi[41] */
    5.94230221045385633407443935898656310894518533675204e-35, /* twoPowerIndex2lo[41] */
  } ,
  {
    1.57598084510788649659218663146020844578742980957031e+00, /* twoPowerIndex2hi[42] */
    -1.01369164712783039807957177429288269249745537889645e-17, /* twoPowerIndex2mi[42] */
    5.43913851556220712785038586461119929989856637311730e-34, /* twoPowerIndex2lo[42] */
  } ,
  {
    1.59314215134226699888131406623870134353637695312500e+00, /* twoPowerIndex2hi[43] */
    -1.00944065423119637216151952902063201612012779755882e-16, /* twoPowerIndex2mi[43] */
    4.60848399034962572477662350836868643017178368092553e-33, /* twoPowerIndex2lo[43] */
  } ,
  {
    1.61049033194925428347232809755951166152954101562500e+00, /* twoPowerIndex2hi[44] */
    2.47071925697978878522451183466139791436957933053447e-17, /* twoPowerIndex2mi[44] */
    1.06968477888935897586507304780358756526593706030066e-33, /* twoPowerIndex2lo[44] */
  } ,
  {
    1.62802742185734783397776936908485367894172668457031e+00, /* twoPowerIndex2hi[45] */
    -6.71295508470708408629558620522800193343463268850872e-17, /* twoPowerIndex2mi[45] */
    1.86124288813399584090278118171650158752835667500490e-33, /* twoPowerIndex2lo[45] */
  } ,
  {
    1.64575547815396494577555586147354915738105773925781e+00, /* twoPowerIndex2hi[46] */
    -1.01256799136747726037875241569662212149731136230039e-16, /* twoPowerIndex2mi[46] */
    -6.73838498803664271467304077725442401461793880458369e-34, /* twoPowerIndex2lo[46] */
  } ,
  {
    1.66367658032673637613640948984539136290550231933594e+00, /* twoPowerIndex2hi[47] */
    5.89099269671309967045155789620226639428173542900082e-17, /* twoPowerIndex2mi[47] */
    2.37785299276765025315795732233641105960161859663372e-33, /* twoPowerIndex2lo[47] */
  } ,
  {
    1.68179283050742900407215074665145948529243469238281e+00, /* twoPowerIndex2hi[48] */
    8.19901002058149652012724391042374107310082144797238e-17, /* twoPowerIndex2mi[48] */
    5.10351519472809316392686812760480457926425713861784e-33, /* twoPowerIndex2lo[48] */
  } ,
  {
    1.70010635371852347752508194389520213007926940917969e+00, /* twoPowerIndex2hi[49] */
    -8.02371937039770024588528464451482959960563128920877e-18, /* twoPowerIndex2mi[49] */
    4.50894675051846528463958043437010583905518288996850e-34, /* twoPowerIndex2lo[49] */
  } ,
  {
    1.71861929812247793414314855908742174506187438964844e+00, /* twoPowerIndex2hi[50] */
    -1.85138041826311098821086356969536380719870481925638e-17, /* twoPowerIndex2mi[50] */
    6.41562962530571009881963439719259893730039269925891e-34, /* twoPowerIndex2lo[50] */
  } ,
  {
    1.73733383527370621735030908894259482622146606445312e+00, /* twoPowerIndex2hi[51] */
    3.16438929929295694659064288262436215220581330791541e-17, /* twoPowerIndex2mi[51] */
    2.46812086524635182684409036079744664142196277491485e-33, /* twoPowerIndex2lo[51] */
  } ,
  {
    1.75625216037329945351075366488657891750335693359375e+00, /* twoPowerIndex2hi[52] */
    2.96014069544887330703087179323550026749650613893620e-17, /* twoPowerIndex2mi[52] */
    1.23348227448930022362949427574612725745479960698002e-33, /* twoPowerIndex2lo[52] */
  } ,
  {
    1.77537649252652118825324123463360592722892761230469e+00, /* twoPowerIndex2hi[53] */
    6.42973179655657203395602172161574258202382771355994e-17, /* twoPowerIndex2mi[53] */
    -3.05903038196122316059732104267589318807211463677414e-33, /* twoPowerIndex2lo[53] */
  } ,
  {
    1.79470907500310716820024481421569362282752990722656e+00, /* twoPowerIndex2hi[54] */
    1.82274584279120867697625715862678123206602563412216e-17, /* twoPowerIndex2mi[54] */
    1.42176433874694971095041068746172287320972847635154e-33, /* twoPowerIndex2lo[54] */
  } ,
  {
    1.81425217550039885594514998956583440303802490234375e+00, /* twoPowerIndex2hi[55] */
    -9.96953153892034881983229632097342495877571562964276e-17, /* twoPowerIndex2mi[55] */
    -5.86224914377491774994695195482808191103744720514440e-33, /* twoPowerIndex2lo[55] */
  } ,
  {
    1.83400808640934243065601094713201746344566345214844e+00, /* twoPowerIndex2hi[56] */
    3.28310722424562720351405816760294702167288526703081e-17, /* twoPowerIndex2mi[56] */
    -6.42508934795304248095271046696049734574532048424330e-34, /* twoPowerIndex2lo[56] */
  } ,
  {
    1.85397912508338547077357816306175664067268371582031e+00, /* twoPowerIndex2hi[57] */
    9.76188749072759353840331670682321086158335176176729e-17, /* twoPowerIndex2mi[57] */
    4.61481577205566482307976345637533484680898060020057e-33, /* twoPowerIndex2lo[57] */
  } ,
  {
    1.87416763411029996255763307999586686491966247558594e+00, /* twoPowerIndex2hi[58] */
    -6.12276341300414256163658402373731493255704994623650e-17, /* twoPowerIndex2mi[58] */
    5.28588559402507397372575432425046614667899080733482e-33, /* twoPowerIndex2lo[58] */
  } ,
  {
    1.89457598158696560730618330126162618398666381835938e+00, /* twoPowerIndex2hi[59] */
    3.40340353521652967060147928999507962708632290832738e-17, /* twoPowerIndex2mi[59] */
    1.72475099549343225430579028439403217279441019556785e-33, /* twoPowerIndex2lo[59] */
  } ,
  {
    1.91520656139714740007207183225546032190322875976562e+00, /* twoPowerIndex2hi[60] */
    -1.06199460561959626376283195555328606320260702029334e-16, /* twoPowerIndex2mi[60] */
    -3.05776975679132548538006102719337626149343902119718e-33, /* twoPowerIndex2lo[60] */
  } ,
  {
    1.93606179349229434727419629780342802405357360839844e+00, /* twoPowerIndex2hi[61] */
    1.03323859606763257447769151803649788699571393339738e-16, /* twoPowerIndex2mi[61] */
    6.05301367682062275405664830597304146844867569493449e-33, /* twoPowerIndex2lo[61] */
  } ,
  {
    1.95714412417540017941064434126019477844238281250000e+00, /* twoPowerIndex2hi[62] */
    8.96076779103666776760155050762912042076490756639488e-17, /* twoPowerIndex2mi[62] */
    -9.63267661361827588458686334472185443533033181828620e-34, /* twoPowerIndex2lo[62] */
  } ,
  {
    1.97845602638795092786949680885300040245056152343750e+00, /* twoPowerIndex2hi[63] */
    4.03887531092781665749784154795462589642365074083484e-17, /* twoPowerIndex2mi[63] */
    3.58120371667786223934924900740488031476290303118010e-34, /* twoPowerIndex2lo[63] */
  } };



